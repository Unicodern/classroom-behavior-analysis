# # 加载预训练模型
# from ultralytics import YOLO

# # 加载预训练模型
# model = YOLO("models\yolov8n-pose.pt")  # 使用YOLOv8 Nano版本作为起点
# # 查看版本信息
# print(model.info())
# print("加载完成")

# result = model('test.png', save=True, show=True)
# result[0].show()


import cv2
img=cv2.imread(r'runs\pose\predict\test.jpg',0)   #转变为灰度图
cv2.imshow('image',img)
k=cv2.waitKey(0)
if k==27:    #按Esc键直接退出
  cv2.destroyAllWindows()
elif k==ord('s'):    #按s键先保存灰度图，再退出
  cv2.imwrite('result.png',img)
  cv2.destroyAllWindows()