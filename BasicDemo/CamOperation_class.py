# -- coding: utf-8 --
import sys
import threading
import msvcrt
import _tkinter
import tkinter.messagebox
from tkinter import * 
from tkinter.messagebox import *
import tkinter as tk
import numpy as np
import cv2
import time
import sys, os
import datetime
import inspect
import ctypes
import random
from PIL import Image,ImageTk
from ctypes import *
from tkinter import ttk
from datetime import datetime
from pathlib import Path

sys.path.append("../MvImport")
from MvCameraControl_class import *

import model_test


def Async_raise(tid, exctype):
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")

def Stop_thread(thread):
    Async_raise(thread.ident, SystemExit)

class CameraOperation():

    def __init__(self,obj_cam,st_device_list,n_connect_num=0,b_open_device=False,b_start_grabbing = False,h_thread_handle=None,\
                b_thread_closed=False,st_frame_info=None,b_exit=False,b_save_bmp=False,b_save_jpg=False,buf_save_image=None,\
                n_save_image_size=0,n_win_gui_id=0,frame_rate=0,exposure_time=0,gain=0,b_detect_object=False):

        self.obj_cam = obj_cam
        self.st_device_list = st_device_list
        self.n_connect_num = n_connect_num
        self.b_open_device = b_open_device
        self.b_start_grabbing = b_start_grabbing 
        self.b_thread_closed = b_thread_closed
        self.st_frame_info = st_frame_info
        self.b_exit = b_exit
        self.b_save_bmp = b_save_bmp
        self.b_save_jpg = b_save_jpg
        self.buf_save_image = buf_save_image
        self.h_thread_handle = h_thread_handle
        self.n_win_gui_id = n_win_gui_id
        self.n_save_image_size = n_save_image_size
        self.b_thread_closed
        self.frame_rate = frame_rate
        self.exposure_time = exposure_time
        self.gain = gain
        self.b_detect_object = b_detect_object

    def To_hex_str(self,num):
        chaDic = {10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f'}
        hexStr = ""
        if num < 0:
            num = num + 2**32
        while num >= 16:
            digit = num % 16
            hexStr = chaDic.get(digit, str(digit)) + hexStr
            num //= 16
        hexStr = chaDic.get(num, str(num)) + hexStr   
        return hexStr

    def Open_device(self):
        if False == self.b_open_device:
            # ch:选择设备并创建句柄 | en:Select device and create handle
            nConnectionNum = int(self.n_connect_num)
            stDeviceList = cast(self.st_device_list.pDeviceInfo[int(nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)).contents
            self.obj_cam = MvCamera()
            ret = self.obj_cam.MV_CC_CreateHandle(stDeviceList)
            if ret != 0:
                self.obj_cam.MV_CC_DestroyHandle()
                tkinter.messagebox.showerror('show error','create handle fail! ret = '+ self.To_hex_str(ret))
                return ret

            ret = self.obj_cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
            if ret != 0:
                tkinter.messagebox.showerror('show error','open device fail! ret = '+ self.To_hex_str(ret))
                return ret
            print ("open device successfully!")
            self.b_open_device = True
            self.b_thread_closed = False

            # ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
            if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
                nPacketSize = self.obj_cam.MV_CC_GetOptimalPacketSize()
                if int(nPacketSize) > 0:
                    ret = self.obj_cam.MV_CC_SetIntValue("GevSCPSPacketSize",nPacketSize)
                    if ret != 0:
                        print ("warning: set packet size fail! ret[0x%x]" % ret)
                else:
                    print ("warning: set packet size fail! ret[0x%x]" % nPacketSize)

            stBool = c_bool(False)
            ret =self.obj_cam.MV_CC_GetBoolValue("AcquisitionFrameRateEnable", stBool)
            if ret != 0:
                print ("get acquisition frame rate enable fail! ret[0x%x]" % ret)

            # ch:设置触发模式为off | en:Set trigger mode as off
            ret = self.obj_cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
            if ret != 0:
                print ("set trigger mode fail! ret[0x%x]" % ret)
            return 0

    def Start_grabbing(self,root,image_panel,video_panel,label_status,label_result):
        if False == self.b_start_grabbing and True == self.b_open_device:
            self.b_exit = False
            ret = self.obj_cam.MV_CC_StartGrabbing()
            if ret != 0:
                tkinter.messagebox.showerror('show error','start grabbing fail! ret = '+ self.To_hex_str(ret))
                return
            self.b_start_grabbing = True
            print ("start grabbing successfully!")
            try:
                self.n_win_gui_id = random.randint(1,10000)
                self.h_thread_handle = threading.Thread(target=CameraOperation.Work_thread, args=(self,root,image_panel,video_panel,label_status,label_result))
                self.h_thread_handle.start()
                self.b_thread_closed = True
            except:
                tkinter.messagebox.showerror('show error','error: unable to start thread')
                False == self.b_start_grabbing

    def Stop_grabbing(self):
        if True == self.b_start_grabbing and self.b_open_device == True:
            #退出线程
            if True == self.b_thread_closed:
                Stop_thread(self.h_thread_handle)
                self.b_thread_closed = False
            ret = self.obj_cam.MV_CC_StopGrabbing()
            if ret != 0:
                tkinter.messagebox.showerror('show error','stop grabbing fail! ret = '+self.To_hex_str(ret))
                return
            print ("stop grabbing successfully!")
            self.b_start_grabbing = False
            self.b_exit  = True      

    def Close_device(self):
        if True == self.b_open_device:
            #退出线程
            if True == self.b_thread_closed:
                Stop_thread(self.h_thread_handle)
                self.b_thread_closed = False
            ret = self.obj_cam.MV_CC_CloseDevice()
            if ret != 0:
                tkinter.messagebox.showerror('show error','close deivce fail! ret = '+self.To_hex_str(ret))
                return
                
        # ch:销毁句柄 | Destroy handle
        self.obj_cam.MV_CC_DestroyHandle()
        self.b_open_device = False
        self.b_start_grabbing = False
        self.b_exit  = True
        print ("close device successfully!")

    def Set_trigger_mode(self,strMode):
        if True == self.b_open_device:
            if "continuous" == strMode: 
                ret = self.obj_cam.MV_CC_SetEnumValue("TriggerMode",0)
                if ret != 0:
                    tkinter.messagebox.showerror('show error','set triggermode fail! ret = '+self.To_hex_str(ret))
            if "triggermode" == strMode:
                ret = self.obj_cam.MV_CC_SetEnumValue("TriggerMode",1)
                if ret != 0:
                    tkinter.messagebox.showerror('show error','set triggermode fail! ret = '+self.To_hex_str(ret))
                ret = self.obj_cam.MV_CC_SetEnumValue("TriggerSource",7)
                if ret != 0:
                    tkinter.messagebox.showerror('show error','set triggersource fail! ret = '+self.To_hex_str(ret))

    def Trigger_once(self,nCommand):
        if True == self.b_open_device:
            if 1 == nCommand: 
                ret = self.obj_cam.MV_CC_SetCommandValue("TriggerSoftware")
                if ret != 0:
                    tkinter.messagebox.showerror('show error','set triggersoftware fail! ret = '+self.To_hex_str(ret))

    def Get_parameter(self):
        if True == self.b_open_device:
            stFloatParam_FrameRate =  MVCC_FLOATVALUE()
            memset(byref(stFloatParam_FrameRate), 0, sizeof(MVCC_FLOATVALUE))
            stFloatParam_exposureTime = MVCC_FLOATVALUE()
            memset(byref(stFloatParam_exposureTime), 0, sizeof(MVCC_FLOATVALUE))
            stFloatParam_gain = MVCC_FLOATVALUE()
            memset(byref(stFloatParam_gain), 0, sizeof(MVCC_FLOATVALUE))
            ret = self.obj_cam.MV_CC_GetFloatValue("AcquisitionFrameRate", stFloatParam_FrameRate)
            if ret != 0:
                tkinter.messagebox.showerror('show error','get acquistion frame rate fail! ret = '+self.To_hex_str(ret))
            self.frame_rate = stFloatParam_FrameRate.fCurValue
            ret = self.obj_cam.MV_CC_GetFloatValue("ExposureTime", stFloatParam_exposureTime)
            if ret != 0:
                tkinter.messagebox.showerror('show error','get exposure time fail! ret = '+self.To_hex_str(ret))
            self.exposure_time = stFloatParam_exposureTime.fCurValue
            ret = self.obj_cam.MV_CC_GetFloatValue("Gain", stFloatParam_gain)
            if ret != 0:
                tkinter.messagebox.showerror('show error','get gain fail! ret = '+self.To_hex_str(ret))
            self.gain = stFloatParam_gain.fCurValue
            tkinter.messagebox.showinfo('show info','get parameter success!')

    def Set_parameter(self,frameRate,exposureTime,gain):
        if '' == frameRate or '' == exposureTime or '' == gain:
            tkinter.messagebox.showinfo('show info','please type in the text box !')
            return
        if True == self.b_open_device:
            ret = self.obj_cam.MV_CC_SetFloatValue("ExposureTime",float(exposureTime))
            if ret != 0:
                tkinter.messagebox.showerror('show error','set exposure time fail! ret = '+self.To_hex_str(ret))

            ret = self.obj_cam.MV_CC_SetFloatValue("Gain",float(gain))
            if ret != 0:
                tkinter.messagebox.showerror('show error','set gain fail! ret = '+self.To_hex_str(ret))

            ret = self.obj_cam.MV_CC_SetFloatValue("AcquisitionFrameRate",float(frameRate))
            if ret != 0:
                tkinter.messagebox.showerror('show error','set acquistion frame rate fail! ret = '+self.To_hex_str(ret))

            tkinter.messagebox.showinfo('show info','set parameter success!')

    def Work_thread(self,root,image_panel,video_panel,label_status,label_result):
        stOutFrame = MV_FRAME_OUT()  
        img_buff = None
        buf_cache = None
        numArray = None
        while True:
            ret = self.obj_cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
            if 0 == ret:
                if None == buf_cache:
                    buf_cache = (c_ubyte * stOutFrame.stFrameInfo.nFrameLen)()
                #获取到图像的时间开始节点获取到图像的时间开始节点
                self.st_frame_info = stOutFrame.stFrameInfo
                cdll.msvcrt.memcpy(byref(buf_cache), stOutFrame.pBufAddr, self.st_frame_info.nFrameLen)
                #print ("get one frame: Width[%d], Height[%d], nFrameNum[%d]"  % (self.st_frame_info.nWidth, self.st_frame_info.nHeight, self.st_frame_info.nFrameNum))
                self.n_save_image_size = self.st_frame_info.nWidth * self.st_frame_info.nHeight * 3 + 2048
                if img_buff is None:
                    img_buff = (c_ubyte * self.n_save_image_size)()
                
                if True == self.b_save_jpg:
                    self.Save_jpg(buf_cache,image_panel) #ch:保存Jpg图片 | en:Save Jpg
                if True == self.b_save_bmp:
                    self.Save_Bmp(buf_cache) #ch:保存Bmp图片 | en:Save Bmp
                if True == self.b_detect_object:
                    self.Detect_object(buf_cache,image_panel,label_status,label_result)
            else:
                print("no data, nret = "+self.To_hex_str(ret))
                continue

            #转换像素结构体赋值
            stConvertParam = MV_CC_PIXEL_CONVERT_PARAM()
            memset(byref(stConvertParam), 0, sizeof(stConvertParam))
            stConvertParam.nWidth = self.st_frame_info.nWidth
            stConvertParam.nHeight = self.st_frame_info.nHeight
            stConvertParam.pSrcData = cast(buf_cache, POINTER(c_ubyte))
            stConvertParam.nSrcDataLen = self.st_frame_info.nFrameLen
            stConvertParam.enSrcPixelType = self.st_frame_info.enPixelType 

            # RGB直接显示
            if PixelType_Gvsp_RGB8_Packed == self.st_frame_info.enPixelType:
                numArray = CameraOperation.Color_numpy(self,buf_cache,self.st_frame_info.nWidth,self.st_frame_info.nHeight)

            #如果是彩色且非RGB则转为RGB后显示
            else:
                nConvertSize = self.st_frame_info.nWidth * self.st_frame_info.nHeight * 3
                stConvertParam.enDstPixelType = PixelType_Gvsp_RGB8_Packed
                stConvertParam.pDstBuffer = (c_ubyte * nConvertSize)()
                stConvertParam.nDstBufferSize = nConvertSize
                time_start=time.time()
                ret = self.obj_cam.MV_CC_ConvertPixelType(stConvertParam)
                time_end=time.time()
                #print('MV_CC_ConvertPixelType:',time_end - time_start) 
                if ret != 0:
                    tkinter.messagebox.showerror('show error','convert pixel fail! ret = '+self.To_hex_str(ret))
                    continue
                cdll.msvcrt.memcpy(byref(img_buff), stConvertParam.pDstBuffer, nConvertSize)
                numArray = CameraOperation.Color_numpy(self,img_buff,self.st_frame_info.nWidth,self.st_frame_info.nHeight)

            #合并OpenCV到Tkinter界面中
            current_image = Image.fromarray(numArray).resize((768, 512), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=current_image, master=root)
            video_panel.imgtk = imgtk       
            video_panel.config(image=imgtk) 
            root.obr = imgtk
            nRet = self.obj_cam.MV_CC_FreeImageBuffer(stOutFrame)
            if self.b_exit == True:
                if img_buff is not None:
                    del img_buff
                if buf_cache is not None:
                    del buf_cache
                break

    def Save_jpg(self,buf_cache,panel):
        if(None == buf_cache):
            return
        self.buf_save_image = None

        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = "HIK_"+  current_time    # 获取当前时间并格式化
        file_path =  os.path.join("Image", filename + ".jpg")
        self.n_save_image_size = self.st_frame_info.nWidth * self.st_frame_info.nHeight * 3 + 2048
        if self.buf_save_image is None:
            self.buf_save_image = (c_ubyte * self.n_save_image_size)()

        stParam = MV_SAVE_IMAGE_PARAM_EX()
        stParam.enImageType = MV_Image_Jpeg;                                        # ch:需要保存的图像类型 | en:Image format to save
        stParam.enPixelType = self.st_frame_info.enPixelType                               # ch:相机对应的像素格式 | en:Camera pixel type
        stParam.nWidth      = self.st_frame_info.nWidth                                    # ch:相机对应的宽 | en:Width
        stParam.nHeight     = self.st_frame_info.nHeight                                   # ch:相机对应的高 | en:Height
        stParam.nDataLen    = self.st_frame_info.nFrameLen
        stParam.pData       = cast(buf_cache, POINTER(c_ubyte))
        stParam.pImageBuffer=  cast(byref(self.buf_save_image), POINTER(c_ubyte)) 
        stParam.nBufferSize = self.n_save_image_size                                 # ch:存储节点的大小 | en:Buffer node size
        stParam.nJpgQuality = 80;                                                    # ch:jpg编码，仅在保存Jpg图像时有效。保存BMP时SDK内忽略该参数
        return_code = self.obj_cam.MV_CC_SaveImageEx2(stParam)            

        if return_code != 0:
            tkinter.messagebox.showerror('show error','save jpg fail! ret = '+self.To_hex_str(return_code))
            self.b_save_jpg = False
            return
        file_open = open(file_path.encode('ascii'), 'wb+')
        img_buff = (c_ubyte * stParam.nImageLen)()
        try:
            cdll.msvcrt.memcpy(byref(img_buff), stParam.pImageBuffer, stParam.nImageLen)
            file_open.write(img_buff)
            self.b_save_jpg = False
            #在面板上显示保存的图片
            self.Show_jpg(file_path,panel)
        except Exception as e:
            self.b_save_jpg = False
            raise Exception("get one frame failed:%s" % e.message)
        if None != img_buff:
            del img_buff
        if None != self.buf_save_image:
            del self.buf_save_image

    def Detect_object(self, buf_cache, panel,label_status,label_result):
        if (None == buf_cache):
            return
        self.buf_save_image = None

        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = "HIK_" + current_time  # 获取当前时间并格式化
        file_path = os.path.join("Image", filename + ".jpg")
        self.n_save_image_size = self.st_frame_info.nWidth * self.st_frame_info.nHeight * 3 + 2048
        if self.buf_save_image is None:
            self.buf_save_image = (c_ubyte * self.n_save_image_size)()

        stParam = MV_SAVE_IMAGE_PARAM_EX()
        stParam.enImageType = MV_Image_Jpeg;  # ch:需要保存的图像类型 | en:Image format to save
        stParam.enPixelType = self.st_frame_info.enPixelType  # ch:相机对应的像素格式 | en:Camera pixel type
        stParam.nWidth = self.st_frame_info.nWidth  # ch:相机对应的宽 | en:Width
        stParam.nHeight = self.st_frame_info.nHeight  # ch:相机对应的高 | en:Height
        stParam.nDataLen = self.st_frame_info.nFrameLen
        stParam.pData = cast(buf_cache, POINTER(c_ubyte))
        stParam.pImageBuffer = cast(byref(self.buf_save_image), POINTER(c_ubyte))
        stParam.nBufferSize = self.n_save_image_size  # ch:存储节点的大小 | en:Buffer node size
        stParam.nJpgQuality = 80;  # ch:jpg编码，仅在保存Jpg图像时有效。保存BMP时SDK内忽略该参数
        return_code = self.obj_cam.MV_CC_SaveImageEx2(stParam)

        if return_code != 0:
            tkinter.messagebox.showerror('show error', 'save jpg fail! ret = ' + self.To_hex_str(return_code))
            self.b_detect_object = False
            return
        file_open = open(file_path.encode('ascii'), 'wb+')
        img_buff = (c_ubyte * stParam.nImageLen)()
        try:
            cdll.msvcrt.memcpy(byref(img_buff), stParam.pImageBuffer, stParam.nImageLen)
            file_open.write(img_buff)
            self.b_detect_object = False
            # 识别图片中的目标
            file_path = self.Predict_jpg(file_path,label_status,label_result)
            self.Show_jpg(file_path, panel)
        except Exception as e:
            self.b_detect_object = False
            raise Exception("get one frame failed:%s" % e.message)
        if None != img_buff:
            del img_buff
        if None != self.buf_save_image:
            del self.buf_save_image

    def Predict_jpg(self,file_path,label_status,label_result):

        # 配置参数，需要跟样本中的classes.txt中顺序一致，区分正反面种类
        CLASS_NAMES = ['bone_front', 'bone_back', \
                       'fish_front', 'fish_back', \
                       'hedgehog_front', 'hedgehog_back', \
                       'heart_front', 'heart_back', \
                       'paw']  # 9种包装类型，4*2+1\n\n
        # 定义映射字典\n
        CLASS_MAPPING = {'bone_front': 'bone', 'bone_back': 'bone', \
                         'fish_front': 'fish', 'fish_back': 'fish', \
                         'hedgehog_front': 'hedgehog', 'hedgehog_back': 'hedgehog', \
                         'heart_front': 'heart', 'heart_back': 'heart', \
                         'paw': 'paw'}
        #不去分正反面的种类
        OBJECT_CLASS_NAMES = ['bone',
                              'fish', \
                              'hedgehog', \
                              'heart', \
                              'paw']

        # 使用预训练模型
        model_path = "packaging_models/best.pt"

        # 初始化检测系统
        try:

            detector = model_test.PackagingDetectionSystem(
                model_path=model_path,
                class_names=CLASS_NAMES,
                class_mapping=CLASS_MAPPING,
                object_class_names=OBJECT_CLASS_NAMES,
                conf_threshold=0.75,
                iou_threshold=0.45
            )
        except Exception as e:
            #logger.error(f"初始化检测系统失败: {str(e)}")
            print(f"初始化检测系统失败: {str(e)}")
            exit(1)

        # 处理单个图像
        # test_image = "测试图像/包装检测1.jpg"
        status, result_path, message = detector.process_image(file_path)
        # logger.info(f"单张图像检测结果: {status}")
        print("包装检测流程完成!"+status + "path i:" +result_path)
        #显示结果到面板上
        self.Update_status(status,label_status)
        self.Update_result(message,label_result)
        #self.Show_jpg(result_path)
        return result_path

    #显示结果ng和ok
    def Update_status(self,signal,label_status):
        if signal == 'NG':
            label_status.config(text='NG', bg='red')
        elif signal == 'OK':
            label_status.config(text='OK', bg='green')


    #显示结果详情
    def Update_result(self,message,label_result):
        label_result.config(text=message)


    def Show_jpg(self,file_path,panel):
        try:
            # 使用 with 语句打开图像文件
            with Image.open(file_path) as img:
                # 调整图片大小
                resized_img = img.resize((900, 600), Image.Resampling.LANCZOS)
                imgtk = ImageTk.PhotoImage(image=resized_img)
                # 在 image_panel 上显示图片
                panel.config(image=imgtk)
                panel.image = imgtk
        except Exception as e:
            tkinter.messagebox.showerror('show error', f'显示图片失败: {str(e)}')
            raise Exception(f"show image error :{str(e)}")




    #显示详细的结果，比如缺少什么，多了什么
    def Show_result(self,signal,label_status):
        if signal == 'ng':
            label_status.config(text='NG', bg='red')
        elif signal == 'ok':
            label_status.config(text='OK', bg='green')

    def Save_Bmp(self,buf_cache):
        if(0 == buf_cache):
            return
        self.buf_save_image = None
        file_path = str(self.st_frame_info.nFrameNum) + ".bmp"    
        self.n_save_image_size = self.st_frame_info.nWidth * self.st_frame_info.nHeight * 3 + 2048
        if self.buf_save_image is None:
            self.buf_save_image = (c_ubyte * self.n_save_image_size)()

        stParam = MV_SAVE_IMAGE_PARAM_EX()
        stParam.enImageType = MV_Image_Bmp;                                        # ch:需要保存的图像类型 | en:Image format to save
        stParam.enPixelType = self.st_frame_info.enPixelType                               # ch:相机对应的像素格式 | en:Camera pixel type
        stParam.nWidth      = self.st_frame_info.nWidth                                    # ch:相机对应的宽 | en:Width
        stParam.nHeight     = self.st_frame_info.nHeight                                   # ch:相机对应的高 | en:Height
        stParam.nDataLen    = self.st_frame_info.nFrameLen
        stParam.pData       = cast(buf_cache, POINTER(c_ubyte))
        stParam.pImageBuffer=  cast(byref(self.buf_save_image), POINTER(c_ubyte)) 
        stParam.nBufferSize = self.n_save_image_size                                 # ch:存储节点的大小 | en:Buffer node size
        return_code = self.obj_cam.MV_CC_SaveImageEx2(stParam)            
        if return_code != 0:
            tkinter.messagebox.showerror('show error','save bmp fail! ret = '+self.To_hex_str(return_code))
            self.b_save_bmp = False
            return
        file_open = open(file_path.encode('ascii'), 'wb+')
        img_buff = (c_ubyte * stParam.nImageLen)()
        try:
            cdll.msvcrt.memcpy(byref(img_buff), stParam.pImageBuffer, stParam.nImageLen)
            file_open.write(img_buff)
            self.b_save_bmp = False
            tkinter.messagebox.showinfo('show info','save bmp success!')
        except:
            self.b_save_bmp = False
            raise Exception("get one frame failed:%s" % e.message)
        if None != img_buff:
            del img_buff
        if None != self.buf_save_image:
            del self.buf_save_image

    def Is_mono_data(self,enGvspPixelType):
        if PixelType_Gvsp_Mono8 == enGvspPixelType or PixelType_Gvsp_Mono10 == enGvspPixelType \
            or PixelType_Gvsp_Mono10_Packed == enGvspPixelType or PixelType_Gvsp_Mono12 == enGvspPixelType \
            or PixelType_Gvsp_Mono12_Packed == enGvspPixelType:
            return True
        else:
            return False

    def Is_color_data(self,enGvspPixelType):
        if PixelType_Gvsp_BayerGR8 == enGvspPixelType or PixelType_Gvsp_BayerRG8 == enGvspPixelType \
            or PixelType_Gvsp_BayerGB8 == enGvspPixelType or PixelType_Gvsp_BayerBG8 == enGvspPixelType \
            or PixelType_Gvsp_BayerGR10 == enGvspPixelType or PixelType_Gvsp_BayerRG10 == enGvspPixelType \
            or PixelType_Gvsp_BayerGB10 == enGvspPixelType or PixelType_Gvsp_BayerBG10 == enGvspPixelType \
            or PixelType_Gvsp_BayerGR12 == enGvspPixelType or PixelType_Gvsp_BayerRG12 == enGvspPixelType \
            or PixelType_Gvsp_BayerGB12 == enGvspPixelType or PixelType_Gvsp_BayerBG12 == enGvspPixelType \
            or PixelType_Gvsp_BayerGR10_Packed == enGvspPixelType or PixelType_Gvsp_BayerRG10_Packed == enGvspPixelType \
            or PixelType_Gvsp_BayerGB10_Packed == enGvspPixelType or PixelType_Gvsp_BayerBG10_Packed == enGvspPixelType \
            or PixelType_Gvsp_BayerGR12_Packed == enGvspPixelType or PixelType_Gvsp_BayerRG12_Packed== enGvspPixelType \
            or PixelType_Gvsp_BayerGB12_Packed == enGvspPixelType or PixelType_Gvsp_BayerBG12_Packed == enGvspPixelType \
            or PixelType_Gvsp_YUV422_Packed == enGvspPixelType or PixelType_Gvsp_YUV422_YUYV_Packed == enGvspPixelType:
            return True
        else:
            return False

    def Mono_numpy(self,data,nWidth,nHeight):
        data_ = np.frombuffer(data, count=int(nWidth * nHeight), dtype=np.uint8, offset=0)
        data_mono_arr = data_.reshape(nHeight, nWidth)
        numArray = np.zeros([nHeight, nWidth, 1],"uint8") 
        numArray[:, :, 0] = data_mono_arr
        return numArray

    def Color_numpy(self,data,nWidth,nHeight):
        data_ = np.frombuffer(data, count=int(nWidth*nHeight*3), dtype=np.uint8, offset=0)
        data_r = data_[0:nWidth*nHeight*3:3]
        data_g = data_[1:nWidth*nHeight*3:3]
        data_b = data_[2:nWidth*nHeight*3:3]

        data_r_arr = data_r.reshape(nHeight, nWidth)
        data_g_arr = data_g.reshape(nHeight, nWidth)
        data_b_arr = data_b.reshape(nHeight, nWidth)
        numArray = np.zeros([nHeight, nWidth, 3],"uint8")

        numArray[:, :, 0] = data_r_arr
        numArray[:, :, 1] = data_g_arr
        numArray[:, :, 2] = data_b_arr
        return numArray