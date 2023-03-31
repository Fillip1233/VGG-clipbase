import os
#<class_name> <left> <top> <right> <bottom> [<difficult>]
#<class_name> <confidence> <left> <top> <right> <bottom>

    # if gtorpred:
    #     path='/mnt/cephfs/dataset/zhenjie/baseline_Xclip/mAP/input/detection-results'
    # else :
    #     path="/mnt/cephfs/dataset/zhenjie/baseline_Xclip/mAP/input/ground-truth"
def savegt(text,name):
    for i in range(8):
        for n in range(8):       
            # os.makedirs('/mnt/cephfs/dataset/zhenjie/baseline_Xclip/mAP/input/ground-truth/b{}frame{}.text'.format(name,i))
            fp=open('/mnt/cephfs/dataset/zhenjie/baseline_Xclip/mAP/input/ground-truth/b{}frame{}{}.txt'.format(name,i,n),"a")
            for m in range(35):
                if text[i][n][m] == 1:
                    fp.writelines('{} 0 0 0 0\n'.format(m))
            fp.close()
def savepred(text,name):
    for i in range(8):
        for n in range(8):       
            fp=open('/mnt/cephfs/dataset/zhenjie/baseline_Xclip/mAP/input/detection-results/b{}frame{}{}.text'.format(name,i,n),"a")
            for m in range(35):
                fp.writelines('{} {} 0 0 0 0/n'.format(m,text[i][n][m]))
            fp.close()
                
        