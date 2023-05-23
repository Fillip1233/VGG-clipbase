import numpy as np
from tqdm import tqdm
from collections import Counter
import os
#处理sub和obj的类

fp=open("/mnt/cephfs/home/alvin/yingqi/STTran/frameclass.txt","r")
objclass=fp.readline()
objclass=objclass.replace('[','')
objclass=objclass.replace(']','')
objclass=objclass.split(',')
# objclass[0]="people"
objclass[-1]=objclass[-1].replace('\n','')
fp.close
fp=open("/mnt/cephfs/home/alvin/yingqi/STTran/framerelation.txt","r")
rlsclass=fp.readline()
rlsclass=rlsclass.replace('[','')
rlsclass=rlsclass.replace(']','')
rlsclass=rlsclass.split(',')
# rlsclass.pop(0)
rlsclass[-1]=rlsclass[-1].replace('\n','')
fp.close

dir_path = "/mnt/cephfs/dataset/zhenjie/agtraindata/frames"

#读取gt数据
def loadgt(path):
    try:
        data=np.load(path,allow_pickle=True).item()
        return data["triples"],data["boxes"]
    except:
        pass
#读取pred
def loadpre(path):
    data=np.load(path,allow_pickle=True).item()
    return data["triples"],data["boxes"],data['rlsscore']

def toclass(triples):
    x=len(triples)
    processed=[]
    for i in range(0,x):
        for z in range(0,3):
            pass
            #print(triples[i][z])
            #print(objclass[int(triples[i][z])])


#f = open("newvideoname.txt","r")
#videoindex=0
#lines = f.readlines()
for i in range(1,1738):
    for z in range(50):
        x=input("next?(y/n)")
        if x=='y':
            print("\n","---------------------------------------------------------------------------")
            try:
                gt_triples,gt_boxes=loadgt('/mnt/cephfs/home/alvin/yingqi/STTran/video.predcls2/video{}/frame{}/gt.npy'.format(i,z))
                sgdet_t,sgdet_box,sgdet_socre=loadpre('/mnt/cephfs/home/alvin/yingqi/STTran/video.sgdet2/video{}/frame{}/with.npy'.format(i,z))
                #img=cv2.imread(dir_path+"/"+lines[videoindex])
                #cv2.imshow("src", img)
                print("gt:\n",toclass(gt_triples))
                print("sgdet:\n",toclass(sgdet_t)[:4])
                #videoindex+=1

            except:
                pass
        else:
            pass