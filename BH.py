from PIL import Image, ImageDraw
import numpy as np
import random
import torch
import numpy as np
import os
from scipy.optimize import basinhopping
import cv2
import torch.nn.functional as Fun

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

letterbox_image = True

threshold = 15
img_transform = lambda i: i


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def add_watermark_to_image(image, xs, watermark):
    rgba_image = image.convert('RGBA')
    rgba_watermark = watermark.convert('RGBA')

    image_x, image_y = rgba_image.size
    watermark_x, watermark_y = rgba_watermark.size
    
     # 旋转角度
    angle = xs[4]
    rgba_watermark = rgba_watermark.rotate(angle, expand=True)
    rgba_watermark.save('rotatewatermark.png')
    
    
    # 缩放图片
    scale = xs[3]
    watermark_scale = min(image_x / (scale * watermark_x), image_y / (scale * watermark_y))
    new_size = (int(watermark_x * watermark_scale), int(watermark_y * watermark_scale))
    # rgba_watermark = rgba_watermark.resize(new_size)
    rgba_watermark = rgba_watermark.resize(new_size, resample=Image.ANTIALIAS)
    # 透明度
    rgba_watermark_mask = rgba_watermark.convert("L").point(lambda x: min(x, int(xs[0])))
    rgba_watermark.putalpha(rgba_watermark_mask)
    


    watermark_x, watermark_y = rgba_watermark.size
    # 水印位置
    # rgba_image.paste(rgba_watermark, (0, 0), rgba_watermark_mask) #右下角
    # 限制水印位置

    a = np.array(xs[1])
    a = np.clip(a, 0, image_x - watermark_x)

    b = np.array(xs[2])
    b = np.clip(b, 0, image_y - watermark_y)

    x_pos = int(a)
    y_pos = int(b)
    rgba_image.paste(rgba_watermark, (x_pos, y_pos), rgba_watermark_mask)  # 右上角
    #rgba_watermark.save('newlogo.png')
    rgba_image.save('rotate.png')
    return rgba_image



threshold = 15

def pic_predict(model, img, path):
    path = path.split('/')[2]
    path = 'result/' + path
    img.save(path)
    
    img = cv2.imread(path)
    img = img_transform(img).astype(np.float32)
    img = torch.from_numpy(np.transpose(img, (2, 0, 1)))
    img = torch.tensor(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    
    outputs = model(img)
    result = Fun.softmax(outputs, dim=-1).cpu().numpy()[0]
    pos = np.argmax(result)
    score = result[pos]
    
    return pos, score



def predict_age(model, img, path, xs, watermark):
    # Perturb the image with the given pixel(s) x and get the prediction of the model
    imgs_perturbed = add_watermark_to_image(img, xs, watermark)
    imgs_perturbed = imgs_perturbed.convert('RGB')
    _, score = pic_predict(model,imgs_perturbed,path)
    
    # This function should always be minimized, so return its complement if needed
    return score


def attack_success(model, img,path,age, xs, watermark, targeted_attack=False, verbose=False):
    # Perturb the image with the given pixel(s) and get the prediction of the model
    attack_image = add_watermark_to_image(img, xs, watermark)
    attack_image = attack_image.convert('RGB')
    result, _ = pic_predict(model,attack_image,path)

    # if verbose:
    #     print('Confidence:', predict[0][target_class])
    if ((targeted_attack and abs(result - age) <= threshold) or
            (not targeted_attack and abs(result - age) > threshold)):
        return True
    
    
def get_watermark_size(watermark, image, scale):
    watermark_x1, watermark_y1 = watermark.size
    image_x, image_y = image.size
    watermark_scale = min(image_x / (scale * watermark_x1), image_y / (scale * watermark_y1))
    # print(watermark_scale)
    watermark_x1 = int(watermark_x1 * watermark_scale)
    watermark_y1 = int(watermark_y1 * watermark_scale)
    return watermark_x1, watermark_y1


def attack(model, im_before, age,path, im_watermark, xs=[100, 0, 0,1.5,45], niter=1):

    def predict_fn(xs):
        return predict_age(model, im_before,path, xs, im_watermark)

    def callback_fn(xs, f, accept):
        return attack_success(model, im_before,path,age, xs, im_watermark, verbose=False)

    class MyTakeStep(object):
        def __init__(self, stepsize=10):
            self.stepsize = stepsize

        def __call__(self, x):
            s = self.stepsize
            x[0] += np.random.uniform(-2 * s, 2 * s)
            x[1] += np.random.uniform(-5 * s, 5 * s)
            x[2] += np.random.uniform(-5 * s, 5 * s)
            x[3] += np.random.uniform(-0.05 * s, 0.05 * s)
            x[4] += np.random.uniform(-5 * s,5 * s)
            scale = x[3]
            angle = x[4]
            # print("change_x: " + str(x))
            return x

    mytakestep = MyTakeStep()

    class MyBounds(object):
        watermark_x, watermark_y = get_watermark_size(im_watermark,im_before,xs[3])
        image_x, image_y = im_before.size
        
        def __init__(self,xmax=[255,image_x-watermark_x,image_y-watermark_y,4,359],xmin=[100, 0, 0,1.5,0]):
            # print("x_max and x_min:" + str(xmax) + str(xmin))
            self.xmax = np.array(xmax)
            self.xmin = np.array(xmin)

        def __call__(self, **kwargs):
            image_x, image_y = im_before.size
            x = kwargs["x_new"]
            # print("new_x: " + str(x))
            watermark_x, watermark_y = get_watermark_size(im_watermark,im_before,x[3])
            xmax=[255,image_x-watermark_x,image_y-watermark_y,4,359]
            xmin=[100, 0, 0,1.5,0]
            self.xmax = np.array(xmax)
            self.xmin = np.array(xmin)
            # print(np.array(xmax),np.array(xmin))
            tmax = bool(np.all(x <= self.xmax))
            tmin = bool(np.all(x >= self.xmin))
            # print(tmax,tmin)
            return tmax and tmin

    mybounds = MyBounds()

    attack_result = basinhopping(func=predict_fn, x0=xs, callback=callback_fn, take_step=mytakestep,
                                 accept_test=mybounds, niter=niter)
    return attack_result




def BH_Calculation(model,path,image_1,age,im_watermark, xs=[100, 0, 0, 1.5,0], niter=1):
    attack_image = add_watermark_to_image(image_1, xs, im_watermark).convert('RGB')
    Optimal_solutions = []
    result_score = None
    predict, score = pic_predict(model,attack_image,path)
    if abs(predict - age) <= threshold:
        result = attack(model, image_1, age,path, im_watermark, xs=xs, niter=niter)
        print(result.x)
        result_img = add_watermark_to_image(image_1, result.x, im_watermark)
        result_img = result_img.convert('RGB')
        _,new_score = pic_predict(model,result_img,path)
        if  score > new_score:
            Optimal_solutions = result.x
            result_score = new_score
            print("Op    " + "old_score: " + str(score) + ", new_score: " + str(new_score) + '\n')
        else:
            Optimal_solutions = xs
            result_score = score
            print("No" + '\n')

    return Optimal_solutions, result_score
