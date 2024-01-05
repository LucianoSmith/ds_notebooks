import serial
import threading
import time
from PyQt5.QtGui import QImage, qGray
from PyQt5.QtCore import QSize, Qt

# TODO !!!
# https://github.com/anabolyc/neje-engraver-tool/blob/master/NejeEngraverApp/Form_Main.cs#L999
# Form_Main.SOFTWARE_TYPE.NOR
# Zoom = 1
MAX_WIDTH = 490
MAX_HEIGHT = 490
# PORT = "/dev/ttyUSB0"
# PORT = "COM4"


class pyGraver():
    def __init__(self, port="/dev/ttyUSB0"):
        self.remain_cmd = None
        self.cmd_todecode = None
        self.check_online = False
        self.askToStop = False
        self.uploadInProcess = False
        self.carvingPercentProgress = 100
        self.numericUpDown_times = 1
        self.fan_RPM = 0
        self.fan_Precent = 0
        self.serial = serial.Serial(port, 57600)
        print("PUERTO : ", port)
        self.location = [MAX_WIDTH//2, MAX_HEIGHT//2]
        if self.serial.isOpen():
            self.init_connexion()
        else:
            print("Error : no serial connection")

    def init_connexion(self):
        # #FF#09#5A#A5
        self.send_CMD(9, 90, 165)
        self.wait_data() # responde CONNECTION OK !
        print("lucho 1")
        print("lucho 2")
        
        # #FF#AA#08#01#01#5A#A5#55
        self.send_CMD_array([170, 8, 1, 1, 90, 165, 85])
        #self.thread = threading.Thread(target=self.connect)
        #self.thread.start()

    def connect(self):
        print("lucho 3")
        cont = True
        while cont:
            self.dataReceivedV2()
            time.sleep(0.02)
            if self.askToStop:
                cont = False

    def close(self):
        self.askToStop = True
        self.serial.close()

    def wait_data(self):
        finish = False
        while not finish:
            finish = self.dataReceived()
            time.sleep(0.02)

    def wait_carving(self):
        while not (self.carvingPercentProgress == 100):
            time.sleep(0.02)

    def send_CMD(self, D1, D2, D3):
        buffer = bytes([255, D1, D2, D3])
        print("send", buffer)
        self.serial.write(buffer)
        self.serial.flush()

    def send_CMD_array(self, arr):
        print("send", arr)
        buffer = bytes([255]+arr)
        print("send", buffer)
        self.serial.write(buffer)
        self.serial.flush()

    def decode(self, D1, D2, D3):
        # print(D1, D2, D3)
        if D1 == 0:
            print("ONLINE")
            self.online = True
            return
        # elif D1 == 1:
        #    break
        elif D1 == 2:
            print("CONNECTION OK !")

    def dataReceived(self):
        nbToRead = self.serial.inWaiting()
        if nbToRead == 0:
            return False
        #
        data = bytearray(self.serial.read(nbToRead))
        #
        if self.remain_cmd is not None:
            self.cmd_todecode = bytearray(self.remain_cmd)
            self.cmd_todecode.extend(data)
        else:
            self.cmd_todecode = bytearray(data)
        self.remain_cmd = None

        if (len(self.cmd_todecode) % 4 != 0):
            self.remain_cmd = bytearray(len(self.cmd_todecode) % 4)
            lr = len(self.remain_cmd)
            lt = len(self.cmd_todecode)
            for i in range(len(self.remain_cmd)):
                self.remain_cmd[i] = self.cmd_todecode[lt - lr + i]
            array2 = bytearray(self.cmd_todecode)
            self.cmd_todecode = bytearray(len(array2) - lr)
            for j in range(len(array2) - lr):
                self.cmd_todecode[j] = array2[j]
        cmd = self.cmd_todecode
        for k in range(len(cmd)):
            if (cmd[k] == 255) and (k+3) < len(cmd):
                self.decode(cmd[k + 1], cmd[k + 2], cmd[k + 3])
        return True

    def dataReceivedV2(self):
        if self.uploadInProcess:
            print("uploadInProcess")
            return False
        nbToRead = self.serial.inWaiting()
        if nbToRead == 0:
            return False
        data = bytearray(self.serial.read_until(b"\x55"))
        data = [data[i] for i in range(len(data))]
        if len(data) < 8:
            return
        if data[0] != 255 or data[1] != 170:
            return
        if data[2] == 8:
            self.readUploadInfo(data)
        elif data[2] == 11:
            self.readStatus(data)
        elif data[2] == 14:
            self.readCarvingProgress(data)
        elif data[2] == 16:
            self.readAfterUpload(data)
        else:
            print(data)

    def readAfterUpload(self, data):
        # receive after upload        --x--  --y--  --w--   --h--
        # [255, 170, 16, 2, 1, 1, 80, 0, 50, 0, 50, 0, 166, 0, 171, 85]
        if len(data) == 16 and data[:6] == [255, 170, 16, 2, 1, 1]:
            x = data[7]*256 + data[8]
            y = data[9]*256 + data[10]
            w = data[11]*256 + data[12]
            h = data[13]*256 + data[14]
            print("readAfterUpload", x, y, w, h)

    def readCarvingProgress(self, data):
        # carving progress          %  --x-- --y--
        # [255, 170, 14, 13, 2, 37, 0, 0, 1, 0, 65, 0, 0, 85]
        # [255, 170, 14, 13, 2, 37, 13, 0, 18, 0, 41, 0, 22, 85]

        if len(data) == 14 and data[:6] == [255, 170, 14, 13, 2, 37]:
            p = data[6]
            x = data[7]*256 + data[8]
            y = data[9]*256 + data[10]
            c = data[11]*256 + data[12]
            print("CarvingProgress", p, x, y, c)
            self.carvingPercentProgress = p

    def readUploadInfo(self, data):
        if data[:5] == [255, 170, 8, 4, 1]:
            if data[5] == 2:
                # [255, 170, 8, 4, 1, 2, 0, 85] # receive image info
                print("receive image info")
            elif data[5] == 3:
                if data[6] == 0:
                    # [255, 170, 8, 4, 1, 3, 0, 85] # waiting for image
                    self.waitingForImage = True
                else:
                    # [255, 170, 8, 4, 1, 3, 8, 85]  # upload en cours (%)
                    self.waitingForImage = False
                    print("Download image percent :", data[6])
            elif data[5] == 4:  # [255, 170, 8, 4, 1, 4, 1, 85] # upload fini
                print("Download image finished !")
                print(data)

    def readStatus(self, data):
        # statut
        # [255, 170, 11, 11, 2, 33, 255, 255, 255, 0, 85] 33 : temp
        if len(data) == 11 and data[:5] == [255, 170, 11, 11, 2]:
            temp = data[5]
            print("temperature :", temp)

    def moveXY(self, x, y):
        self.send_CMD_array([
            170, 16, 5, 1, 80, 1,
            int(x // 256), int(x % 256),
            int(y // 256), int(y % 256),
            0, 0, 0, 0, 85])

    def luchon(self):
        print("lucho self")
        
    def move(self):
        x = self.location[0]
        y = self.location[1]
        self.moveXY(x, y)

    def moveUp(self):
        self.location[1] = max(0, self.location[1]-4)
        self.move()

    def moveDown(self):
        self.location[1] = min(MAX_HEIGHT, self.location[1]+4)
        self.move()

    def moveLeft(self):
        self.location[0] = max(0, self.location[0]-4)
        self.move()

    def moveRight(self):
        self.location[0] = min(MAX_WIDTH, self.location[0]+4)
        self.move()

    def showWindow(self, x, y, w, h):
        self.send_CMD_array([
            170, 16, 5, 1, 80, 2,
            int(x // 256), int(x % 256),
            int(y // 256), int(y % 256),
            int(w // 256), int(w % 256),
            int(h // 256), int(h % 256),
            85])

    def stopShowWindow(self):
        self.send_CMD_array([170, 16, 5, 1, 80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 85])

    # laser power + idle (au repos ?)
    def setPWD(self, p, i):
        # TODO à tester !!
        # print("setPWD", p, i)
        self.send_CMD_array([170, 11, 3, 1, 15, p, i, 0, 0, 85])

    def stopCarving(self):
        print("stopCarving")
        self.send_CMD_array([170, 8, 2, 1, 1, 2, 85])
        # stop carving
        # FF#AA#08#02#01#01#02#55

    
        # comienza con 
        # FF 255
        # AA 170
        # 08 8
        # 02 2
        # 01 1
        # 01 1
        # 02 2
        # 55 85

    def pauseCarving(self):
        print("pauseCarving")
        self.send_CMD_array([170, 8, 2, 1, 1, 0, 85])


        # pause carving
        # FF#AA#08#02#01#01#00#55

    def sendImage(self, image, x=50, y=50):
        print("sendImage")
        w = image.width()
        h = image.height()
        # TODO check size < max with/heigt
        wr = 8 * ((w + 7) // 8)  #
        le = (wr * h) // 8
        # send image information
        self.send_CMD_array([
            170, 22, 4, 2, 1, 80,
            int(x // 256), int(x % 256),
            int(y // 256), int(y % 256),
            int(wr // 256), int(wr % 256),
            int(h // 256), int(h % 256),
            0, 0,
            int(le // 256), int(le % 256),
            int(w // 256), int(w % 256),
            85])
        time.sleep(1)

        # rezise image to wr *h with with pixel on right :
        img_big = image.scaled(QSize(wr, h))
        img_big.fill(Qt.white)
        # miror every other line
        for l in range(h):
            for c in range(w):
                if l % 2:
                    co = w - 1 - c
                else:
                    co = c
                img_big.setPixel(co, l, image.pixel(c, l))
        # transform image as bytearray
        byte_array = bytearray()
        for l in range(h):
            byte_temp = 0
            for c in range(wr):
                if (c > 0) and not (c % 8):
                    byte_array.append(byte_temp)
                    byte_temp = 0
                if qGray(img_big.pixel(c, l)) < 128:
                    byte_temp += 1 << (7-(c % 8))

            byte_array.append(byte_temp)
        byte_array.append(85)
        self.transmit(byte_array, 8192)

    def transmit(self, data, chunkSize):
        print("transmit")
        self.uploadInProcess = True
        self.serial.write(data)
        self.serial.flush()
        # TODO transmit by chunk ?
        # for i in range(0, data.size(), chunkSize):
        #    print("transmit", i, chunkSize)
        #    self.serial.write(data[i:i+chunkSize])
        #    self.serial.flush()
        print("transmit finish")
        self.uploadInProcess = False


if __name__ == '__main__':
    # TODO command line usage
    g = pyGraver("/dev/ttyUSB0")
    time.sleep(2)
    g.setPWD(70, 70)  # TODO value ???
    # time.sleep(2)
    # print("read image")
    image = QImage("logo.png")
    g.showWindow(50, 50, image.width(), image.height())
    time.sleep(3)
    g.stopShowWindow()
    time.sleep(5)
    g.sendImage(image, 50, 150)
    g.wait_carving()
    g.close()
