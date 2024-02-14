
import cv2
import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import transforms
from PIL import Image

ANCHO_IMAGENES = 144 #150
ALTO_IMAGENES = 144 #150

class InsectClassifier(torch.nn.Module):
    def __init__(self, output_units):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding='same') # 144 x 144
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2) # 72 x 72
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding='same')
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2) # 36 x 36
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding='same')
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2) # 18 x 18
        self.conv4 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding='same')
        self.pool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2) # 9 x 9
        self.fc1 = torch.nn.Linear(in_features=128*9*9, out_features=512)       ## MODIF
        #output_units = 1 ## MODIF 
        self.fc2 = torch.nn.Linear(in_features=512, out_features=output_units)  

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))
        x = self.pool4(torch.relu(self.conv4(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        #return x
        return torch.sigmoid(x) ## MODIF         

# Definir la transformación de datos para el conjunto de prueba
transform = transforms.Compose([
    transforms.Resize((ANCHO_IMAGENES, ALTO_IMAGENES)),
    transforms.ToTensor(),
])

def preprocesar_imagen(array_img):
    transformacion = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(ANCHO_IMAGENES, ALTO_IMAGENES)),
        torchvision.transforms.ToTensor(),
        #torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    imagen = Image.fromarray(array_img)
    #imagen = Image.open(imagen_path)
    imagen = transformacion(imagen)
    imagen = imagen.unsqueeze(0)  # Agregar dimensión batch
    return imagen

def predecir_imagen(array_img, modelo):
    
    imagen = preprocesar_imagen(array_img)

    with torch.no_grad():
        salida = modelo(imagen)
    output = salida.numpy()
    
    c = (max(output[0][0], output[0][1])/(output[0][0] + output[0][1]))

    _, indice_prediccion = torch.max(salida, 1)
    
    return indice_prediccion.item(), c

class dsia():

    def __init__(self, model) -> None:
        self.set_params()
        # inicializa el modelo de IA
        self.model = InsectClassifier(self.CANTIDAD_CLASES)
        # Cargar los pesos entrenados
        self.model.load_state_dict(torch.load(model)) # '01_nn1_modelo_insectos2000.pth'
        self.model.eval()  # Establecer el modelo en modo de evaluación

    def set_img(self, img) -> None:
        self.img_orig = img.copy()

    def set_params(self) -> None:
        
        self.rect = 150             # VERIFICAR
        self.cross = 30
        self.umbral = 0.5           # si la confianza no supera el umbral la considera hembra
        self.columns = ['obj_id', 'pixels' ,'x', 'y', 'sex', 'state', 'targetX', 'targetY', 'conf']
        self.drosophila = pd.DataFrame(columns=self.columns) # crea el dataframe
        self.path = pd.DataFrame(columns=['obj_id', 'targetX', 'targetY'])                                                   # path para el laser
        self.path.loc[len(self.path.index)] = [-2, 100, 100] # agrega la posicion del laser 
        self.font = cv2.FONT_HERSHEY_SIMPLEX    
        self.CANTIDAD_CLASES = 2
        self.ANCHO_IMAGENES = self.rect
        self.ALTO_IMAGENES = self.rect
        self.draw_bbox = 0  # draw bounding boxes
        self.draw_objn = 0  # draw object number
        self.draw_conf = 0  # draw confidence
        self.draw_targ = 0  # draw target
        self.draw_rout = 0  # draw route 

    def increase_brightness(self, img, value = 30):
        # incrementa el brillo de la imagen
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

        return img
    
    def image_threshold(self, img, br = 150, gb = 3, li = 230, ls= 255):
        # binarizacion, br cantidad de brillo, gb = kernel del gaussian blur, li y ls limites superior e inferior del theadhold
        img_b = self.increase_brightness(img, value=br)
        img_b  = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
        img_b = cv2.GaussianBlur(img_b, (gb,gb), 0)
        _, img_c = cv2.threshold(img_b, li, ls, cv2.THRESH_BINARY)
        return img_c
    
    def image_erodil(self, img, d_iter = 10, e_iter = 10, k = 3):
        # Morphological Operations: erosiona y dilata ... k tamano del kernel, d_iter, e_iter iteraciones en dilatacion y erosion
        # erossion and dilatation
        kernel = np.ones((k,k), np.uint8)

        img = cv2.dilate(img, kernel, iterations = d_iter)
        img = cv2.erode(img, kernel, iterations = e_iter)
        return  img
    
    def image_overlap(self, img):
        # VERIFICAR
        temp_cont, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ret = 'sin clasificar'
        if (len(temp_cont)>1):                      # cuenta contornos en temp_img
            ret = 'overlaped'

        return ret
    
    def image_addBoundingbox(self, img, x0, y0, ov, text, q, y_text = 10, s = "n/a", conf = 0.0):
        
        c = (0,0,0)
        
        if ov == "sin clasificar": c = (0,128,0)        # verde
        if ov == "overlaped": c = (255,0,0)             # rojo
        if ov == "omitido": c = (95,95,95)              # gris
        if ov == "fuera umbral 1": c = (0,0,0)          # negro
        if ov == "fuera umbral 2": c = (255,255,0)      # marron
        if ov == "classified": 
            if s == "f": c = (255,0,255)                # rosa
            if s == "m": c = (0,0,255)                  # azul
        
        if self.draw_bbox==1:
            cv2.rectangle(img, (x0, y0), (x0+self.rect, y0+self.rect), c, 5)

        if text and self.draw_objn==1:
            cv2.putText(img, str(q), (x0, y0-y_text), self.font, 1, c, 2, cv2.LINE_AA)

        if conf>0 and self.draw_conf==1:
            cv2.putText(img, f"{conf:.2f}", (x0+100, y0-y_text), self.font, .8, c, 2, cv2.LINE_AA)

        return img

    def image_contours(self, img_t, img_o, writePNG = False, t = 10, rect = 150, pMin = 4, pMax = 20):
        # detecta y analiza contornos, t? rect: tamanno del recuadro, pMin, pMax: % maximo y minimo de pixes ocupados    
        img_t = cv2.bitwise_not(img_t)
        contornos, _ = cv2.findContours(img_t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        img_fin = img_o.copy()

        hmin = 20
        wmin = 20
        hmax = 120
        wmax = 120
        y_text = 10

        cont = 0
        p = 0
        for c in contornos:
            if len(c)>t:                                # que carajos es t??? y este proc???

                (x, y, w, h) = cv2.boundingRect(c)                          # realiza el bounding box

                x0 = int((x + w/2) - (rect/2))                              # localiza la coordenada x0 basado en el tamaño del recuadro
                y0 = int((y + h/2) - (rect/2))                              # localiza la coordenada y0 basado en el tamaño del recuadro
                
                cont = cont + 1                                             # incrementa la cantidad de objetos encontrados
                
                if ((h>hmin) and (w>wmin)) and ((h<hmax) and (w<wmax)):     # tamaño minimo y maximo del recuadro

                    roi1 = img_t[y0:y0+rect, x0:x0+rect]                    # roi de imagen binarizada
                    q = cv2.countNonZero(roi1)                              # calculo de % de pixels ocupados
                    p = round((q/(rect*rect))*100, 1)
                    
                    if (p>pMin) and (p<pMax):                               # seleccion de % minimo y maximo

                        ov = self.image_overlap(roi1)                            # verifica superposición de imagenes
                    
                        if writePNG:
                            roi = img_o[y0:y0+rect, x0:x0+rect] 
                            cv2.imwrite(str(cont)+'.png', roi)
                        
                    else:

                        ov = "fuera umbral 1"
                    
                else:

                    ov = "fuera umbral 2"
                

                img_fin = self.image_addBoundingbox(img_fin, x0, y0, ov, True, cont, y_text)

                self.drosophila.loc[len(self.drosophila.index)] = [cont, p, x0, y0, "n/a", ov, 0, 0, 0.0]   # registra el roi


        return img_fin    

    def roi_target(self, roi, img, x0, y0, cross = 30):
        # marca el centro de masa, cross tamaño de la mira
        
        c = int(cross/2)                         #  la cruz tiene 2*cross de lado
        
        eyes_lo=np.array([90,60,60])
        eyes_hi=np.array([125,85,85])

        mask=cv2.inRange(roi, eyes_lo, eyes_hi)
        M = cv2.moments(mask)
    
        # calculate x,y coordinate of center
        x = int(M["m10"] / M["m00"])
        y = int(M["m01"] / M["m00"])
    
        x = x0 + x
        y = y0 + y
    
        #cv2.putText(img, "target", (x - 25, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        cv2.circle(img, (x, y), 5, (255, 255, 255), -1)
        cv2.line(img, (x, y-c), (x, y+c), (255, 255, 0), 2) 
        cv2.line(img, (x-c, y), (x+c, y), (255, 255, 0), 2) 

        return x, y, img
    
    def roi_target_plus(self, roi, img, x0, y0, cross = 30):
        # marca el centro de masa, cross tamaño de la mira (teniendo en cuenta que son boundingboxes con errores)
        c = int(cross/2)                         #  la cruz tiene 2*cross de lado

        roi = cv2.bitwise_not(roi)

        kernel = np.ones((3,3), np.uint8)
        roi = cv2.erode(roi, kernel, iterations = 10)

        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)     #AGREGADO
        contornos, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) ### ACA ERROR!!!!
                       


        for contor in contornos:
            (x, y, w, h) = cv2.boundingRect(contor) 
            x = int(x + w/2) + x0
            y = int(y + h/2) + y0

            # put text and highlight the center
            cv2.circle(img, (x, y), 5, (255, 255, 255), -1)    
            cv2.line(img, (x, y-c), (x, y+c), (255, 255, 0), 2) 
            cv2.line(img, (x-c, y), (x+c, y), (255, 255, 0), 2) 

            self.path.loc[len(self.path.index)] = [-1, x, y] 

        return x, y, img
    
    def xyRepair(self, x0, y0, rect = 150):
    
        # faltaria cubrir si se pasa de la imagen, es decir si x+150 > ancho de la imagen
        x1 = x0
        y1 = y0
        x2 = x0 + rect
        y2 = y0 + rect

        if ((x0<0) or (y0<0)):
            if (y0<0):
                y1 = 0
                y2 = rect+y0
            if (x0<0):
                x1 = 0
                x2 = rect+x0

        return x1,y1,x2,y2

    def img_ramdom_classify(self, img):
        sex = random.choice(["f", "m"])
        conf = 1.00
        return sex, conf
    
    def img_yolonas_classify(img):
        pass
        #    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #    cv2.imwrite('temp.png' , img)
        #    pa = 'temp.png'
        #    p = best_model.predict(pa, max_predictions=1, fuse_model=False)
        #    conf = p.prediction.confidence[0]
        #    if (p.prediction.labels[0]==1) or (conf<=0.65): 
        #        sex = 'f' 
        #    else: 
        #        sex = 'm'
        #    print(sex, conf)
        #    return sex, conf

    def img_nn_classify(self, img):
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        p, c = predecir_imagen(img, self.model)

        if (p==0) or (c<=self.umbral): 
            sex = 'f' 
        else: 
            sex = 'm'
        print(sex, c)
        return sex, c

    def img_classification(self, img):
        #sex, conf = self.img_ramdom(img)
        sex, conf = self.img_nn_classify(img)
        return sex, conf

    def image_classification(self, img_de):

        # recorre el dataset de las que estan en estado "sin clasificar"
        img_f = self.img_orig.copy()

        for _, row in self.drosophila.iterrows():

            x0 = int(row.x)
            y0 = int(row.y)

            roi = self.img_orig[y0:y0+self.rect, x0:x0+self.rect]  
            st = row.state

            s = 'n/a'
            conf = 0.0
            if (row.state == 'sin clasificar'):
                s, conf = self.img_classification(roi)                                            
                st = 'classified'
                
            # actualiza el dataframe
            self.drosophila.loc[self.drosophila['obj_id'] == row.obj_id, ['state']] = st
            self.drosophila.loc[self.drosophila['obj_id'] == row.obj_id, ['sex']] = s
            self.drosophila.loc[self.drosophila['obj_id'] == row.obj_id, ['conf']] = conf
            
            # redibuja boundingboxes
            self.img_f = self.image_addBoundingbox(img_f, x0, y0, st, True, row.obj_id, 10, s, conf)

        return self.img_f

    def insect_targeting(self, img):

        img_f = self.img_orig.copy()

        for _, row in self.drosophila.iterrows():

            if (row.targetX>0) or (row.targetY>0): 
                # dibuja directamente
                c = int(self.cross/2)  
                cv2.circle(img_f, (targetX, targetY), 5, (255, 255, 255), -1)        
                cv2.line(img_f, (targetX, targetY-c), (targetX, targetY+c), (255, 255, 0), 2) 
                cv2.line(img_f, (targetX-c, targetY), (targetX+c, targetY), (255, 255, 0), 2) 
                if (self.path[self.path.obj_id == row.obj_id].obj_id.count()==0):
                     self.path.loc[len(self.path.index)] = [row.obj_id, row.targetX, row.targetY] 

            else:
                # calcula y dibuja
                targetX = 0 
                targetY = 0 

                x0 = int(row.x)
                y0 = int(row.y)
                roi = self.img_orig[y0:y0+self.rect, x0:x0+self.rect]  
                conf = 0.0                

                if (row.sex=="f"):                                                                       # elimina hembras 
                    x1, y1, x2, y2 = self.xyRepair(x0, y0, self.rect)
                    roi = self.img_orig[y1:y2, x1:x2]  
                    targetX, targetY, img_f = self.roi_target(roi, img_f, x1, y1, 30)                    # encuentra punto de target de moscas ok
                    self.path.loc[len(self.path.index)] = [row.obj_id, targetX, targetY]                 # registra el target

                if (row.state != "classified"):                                                          # elimina la resto (no machos)
                    x1, y1, x2, y2 = self.xyRepair(x0, y0, self.rect)
                    roi = img[y1:y2, x1:x2]         
                    targetX, targetY, img_f = self.roi_target_plus(roi, img_f, x1, y1, 30)    # encuentra punto de target de moscas con problemas

                # actualiza el dataframe
                self.drosophila.loc[self.drosophila['obj_id'] == row.obj_id, ['targetX']] = targetX
                self.drosophila.loc[self.drosophila['obj_id'] == row.obj_id, ['targetY']] = targetY
                self.drosophila.loc[self.drosophila['obj_id'] == row.obj_id, ['conf']] = conf
            
        return img_f

    def insect_detection(self):

        # detecta los insectos (ubicacion y bounding boxes)
        img_pre = self.image_threshold(self.img_orig, 150, 3, 230, 255)
        img_de = self.image_erodil(img_pre, 10, 10, 3)
        img_c = self.image_contours(img_de, self.img_orig, False)
        return img_c, img_de

    def no_optimization(self, img_f):

        # rutea según el orden del id
        x0 = xf = int(self.path.targetX.iloc[0])
        y0 = yf = int(self.path.targetY.iloc[0])

        for _, row in self.path.iterrows():
            
            x1 = int(row.targetX)
            y1 = int(row.targetY)
            
            cv2.line(img_f, (x0, y0), (x1, y1), (255, 255, 0), 5) 
            x0 = x1
            y0 = y1
            
        cv2.line(img_f, (x0, y0), (xf, yf), (255, 255, 0), 5)

        return img_f

    def route(self):

        # Eliminar duplicados 
        self.path = self.path.drop_duplicates(['targetX','targetY'])
        img = self.no_optimization(self.img_f)
        return img
        
    def draw_target(self, img, x, y):

        c = int(self.cross/2)   
    
        cv2.circle(img, (x, y), 5, (255, 255, 255), -1)
        cv2.line(img, (x, y-c), (x, y+c), (255, 255, 0), 2) 
        cv2.line(img, (x-c, y), (x+c, y), (255, 255, 0), 2) 

        return img

    def draw_insect_info(self):

        # recorre el dataset y redibuja los bounding boxes
        img = self.img_orig.copy()
                             
        for _, row in self.drosophila.iterrows():
            if row.state=="classified":
                img = self.image_addBoundingbox(img, row.x, row.y, row.state, True, row.obj_id, 10, row.sex, row.conf)
            else:
                img = self.image_addBoundingbox(img, row.x, row.y, row.state, True, row.obj_id)

            if self.draw_targ and (row.targetX>0 or row.targetY>0):
                img = self.draw_target(img, row.targetX, row.targetY)        

        if self.draw_rout and (self.path.obj_id.count()>0):
            img = self.route()
        

        return img

    def change_insect(self, id, data):
        
        # cambia un parámetro del insecto según el id y el data
        if (data=='m'):
            self.drosophila.loc[self.drosophila.obj_id==id, "sex"] = 'm'
        elif (data=='f'):
            self.drosophila.loc[self.drosophila.obj_id==id, "sex"] = 'f'
        else:
            self.drosophila.loc[self.drosophila.obj_id==id, "state"] = data
        
        # redibuja el canvas
        img = self.draw_insect_info()
        return img
