import os, sys
sys.path.insert(0,'..')
try:
    import dota_utils as util
except:
    import dota_kit.dota_utils as util
srcpath1 = '/mnt/lustre/yanhongchang/project/one-rpn/mmdetection/demo/work_dirs/out_img/faster_x101/crop512/txt2'
srcpath2 = '/mnt/lustre/yanhongchang/project/one-rpn/mmdetection/demo/work_dirs/out_img/faster_x101/crop1024/txt2'
srcpath3 = '/mnt/lustre/yanhongchang/project/one-rpn/mmdetection/demo/work_dirs/out_img/faster_x101/origin/txt2'
dstpath = '/mnt/lustre/yanhongchang/project/one-rpn/mmdetection/demo/work_dirs/out_img/faster_x101/tr_merge3/txt'
def osp(savepath):
    if not os.path.exists(savepath):
        os.makedirs(savepath)
osp(dstpath)
filelist = util.GetFileFromThisRootDir(srcpath1)
for filename in filelist:
    name = util.custombasename(filename)
    print(name)
    file_1 = open(os.path.join(srcpath1, name + '.txt'), 'r')
    file_2 = open(os.path.join(srcpath2, name + '.txt'), 'r')
    file_3 = open(os.path.join(srcpath3, name + '.txt'), 'r')
    file_new = open(os.path.join(dstpath, name + '.txt'), 'w')

    list1 = []
    for line in file_1.readlines():
        ss = line.strip()
        list1.append(ss)
    file_1.close()

    list2 = []
    for line in file_2.readlines():
        ss = line.strip()
        list2.append(ss)
    file_2.close()

    list3 = []
    for line in file_3.readlines():
        ss = line.strip()
        list3.append(ss)
    file_3.close()

    for i in range(len(list1)):
        file_new.write(list1[i]+'\n')
    for i in range(len(list2)):
        file_new.write(list2[i]+'\n')
    for i in range(len(list3)):
        file_new.write(list3[i]+'\n')
    file_new.close()