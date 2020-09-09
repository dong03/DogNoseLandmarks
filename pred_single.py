import cv2
from tools.transforms import direct_val
import torch
from models.classifier import LandmarksRegressor
import re


class Pred_Single(object):
    def __init__(self,model_path,device,half):
        self.model = LandmarksRegressor(encoder="tf_efficientnet_b7_ns")
        self.transforms = direct_val
        self.model_path = model_path
        self.device = device
        self.half = half

    def predict(self,img_raw,save_path=None):
        #输入为cv2 BGR格式
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        inputs = self.transforms([img_rgb],380)
        inputs = inputs.to(self.device)
        inputs = inputs.unsqueeze(0)
        with torch.no_grad():
            if self.half:
                output = self.model(inputs.half())
            else:
                output = self.model(inputs)
        pred_landmarks = output[0].cpu().numpy()

        if save_path is not None:
            self.save_draw(img_raw,pred_landmarks,save_path)
        return pred_landmarks

    def save_draw(self,img_raw,pred_landmarks,save_path):
        font = cv2.FONT_HERSHEY_SIMPLEX
        for ix,each in enumerate(pred_landmarks):
            cv2.circle(img_raw,(each[0],each[1]),2,(0,0,255),-1)
            cv2.putText(img_raw, str(ix + 1), (each[0],each[1]), font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imwrite(save_path,img_raw)

    def initmodel(self):
        print("loading state dict %s"%self.model_path)
        try:
            checkpoint = torch.load(self.model_path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)
            self.model.load_state_dict({re.sub("^module.", "", k): v for k, v in state_dict.items()}, strict=False)
        except:
            print("error loading %s"%self.model_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        if self.half:
            self.model = self.model.half()
        del checkpoint

if __name__ == '__main__':
    pred_single = Pred_Single(
        model_path='/data/dongchengbo/code/dfdc_1st_Vdcb/weights/best/fix_lr0.01_decay0.8_resume2DeepFakeClassifier_tf_efficientnet_b7_ns_0_last',
        device=torch.device('cuda'),
        half=True
    )
    pred_single.initmodel()

    img = cv2.imread("/data/dongchengbo/code/Sanders_0300.png")
    pred_landmarks = pred_single.predict(img)
    print("landmarks: \n",pred_landmarks)