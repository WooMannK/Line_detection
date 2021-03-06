# -*- coding: utf-8 -*-

import numpy as np                            # 배열 연산 처리에 유용한 모듈
import cv2
import random
import math                      # 파이썬에서 openCV를 사용하기 위한 모듈, 난수 생성 모듈, 수학적 계산 관련 모듈
import time
import pandas as pd             # 데이터 프레임 처리 모듈
import sys  # 파이썬 인터프리터가 제공하는 변수와 함수를 직접 제어할 수 있게 해주는 모듈
import os
from bayes_opt import BayesianOptimization

# 선언 및 초기화
Width = 640                                 # 영상 제원 : 640 x 480 / 30FPS
Height = 480
Offset = 380                                # 이번 프로젝트에서 주어진 Offset 위치 약간 변경함
Gap = 40
line_temp = []                         # left_pos, right_pos 저장 배열

# 실행을 위해 추가한 코드
cap = cv2.VideoCapture("./Line_detection/subProject.avi")
#######

# canny edge img, lines를 인자로 받아 cv2.line으로 직선을 그려주는 함수
def draw_lines(img, lines):
    global Offset
    for line in lines:
        x1, y1, x2, y2 = line[0]
        color = (random.randint(0, 255), random.randint(
            0, 255), random.randint(0, 255))
        img = cv2.line(img, (x1, y1+Offset), (x2, y2+Offset), color, 2)
    return img

# draw rectangle
# img, lpos, rpos, offset을 입력받아 해당 offset라인의 사각형 4개 출력
def draw_rectangle(img, lpos, rpos, offset=0):
    center = int((lpos + rpos) / 2)

    cv2.rectangle(img, (lpos - 5, 15 + offset),         # 왼쪽 lpos 사각형 그림
                       (lpos + 5, 25 + offset),
                       (0, 255, 0), 2)
    cv2.rectangle(img, (rpos - 5, 15 + offset),         # 오른쪽 lpos 사각형 그림
                       (rpos + 5, 25 + offset),
                       (0, 255, 0), 2)
    cv2.rectangle(img, (center-5, 15 + offset),         # lpos, rpos의 중간 좌표에 대한 사각형 그림
                       (center+5, 25 + offset),
                       (0, 255, 0), 2)
    cv2.rectangle(img, (315, 15 + offset),              # Width/2 부분의 영상 중심 사각형 그림
                       (325, 25 + offset),
                       (0, 0, 255), 2)
    return img



# left lines, right lines
# lines를 인자 값으로 받아 왼쪽 차선인지 오른쪽 차선인지 판단하는 함수
def divide_left_right(lines):
    global Width

    # 해당 기울기 사이에 있어야 line이라 판단하는 임계값
    low_slope_threshold = 0
    high_slope_threshold = 9

    # calculate slope & filtering with threshold
    slopes = []                                         # 임계값에 의해 필터링된 기울기와 line 값
    new_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]

        if x2 - x1 == 0:                                # x 좌표가 동일하다면 기울기를 0으로 판단
            slope = 0
        else:
            # 기울기 a = (y2-y1) / (x2-x1)
            slope = float(y2-y1) / float(x2-x1)

        if abs(slope) > low_slope_threshold and abs(slope) < high_slope_threshold:  # 해당 임계값 사이의 기울기와 라인만 저장
            slopes.append(slope)
            new_lines.append(line[0])

    # divide lines left to right
    left_lines = []
    right_lines = []

    for j in range(len(slopes)):
        Line = new_lines[j]
        slope = slopes[j]

        x1, y1, x2, y2 = Line

        if (slope < 0) and (x2 < Width/2 - 90):
            left_lines.append([Line.tolist()])
        elif (slope > 0) and (x1 > Width/2 + 90):
            right_lines.append([Line.tolist()])

    return left_lines, right_lines

# get average m, b of lines
# 여러 line을 인자값으로 받아 모든 라인의 평균값을 대표하는 기울기와 절편 값으로 설정
def get_line_params(lines):
    # sum of x, y, m
    x_sum = 0.0
    y_sum = 0.0
    m_sum = 0.0

    size = len(lines)
    if size == 0:
        return 0, 0

    for line in lines:
        x1, y1, x2, y2 = line[0]

        x_sum += x1 + x2
        y_sum += y1 + y2
        m_sum += float(y2 - y1) / float(x2 - x1)

    x_avg = x_sum / (size * 2)
    y_avg = y_sum / (size * 2)
    m = m_sum / size
    b = y_avg - m * x_avg

    return m, b


# get lpos, rpos
def get_line_pos(img, lines, left=False, right=False):      # 좌,우 line의 위치 반환
    global Width, Height
    global Offset, Gap

    m, b = get_line_params(lines)                           # 기울기 절편을 받아옴

    if m == 0 and b == 0:                                   # 인식되지 않았다면 left는 pos를 0으로, right는 pos를 640으로
        if left:
            pos = 0
        if right:
            pos = Width
    else:
        y = Gap / 2
        pos = (y - b) / m

        b += Offset
        x1 = (Height - b) / float(m)
        x2 = ((Height/2) - b) / float(m)

        cv2.line(img, (int(x1), Height),
                 (int(x2), int(Height/2)), (255, 0, 0), 1)

    return img, int(pos)


# show image and return lpos, rpos
def process_image(frame):                           # 입력받은 프레임당의 이미지에 대한 연산 함수
    global Width
    global Offset, Gap

    # gray
    # BGR 영상을 GRAY로 변환. 이번 프로젝트에는 GRAY로 해야하나?
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # blur
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(
        gray, (kernel_size, kernel_size), 0)    # 가우시안 블러 (5,5)의 값 적용

    # canny edge
    # 다른 엣지와 가까운 곳에서 엣지인지 아닌지를 판단하는 임계값
    low_threshold = 70
    # 사람이 생각하는 직관적인 임계값.
    high_threshold = 163
    # 만약 해당 파라미터 조정이 필요하다만 high를 먼저, low를 나중에
    edge_img = cv2.Canny(np.uint8(blur_gray), low_threshold, high_threshold)

    #cv2.imshow('edge',edge_img)

    # HoughLinesP
    # Offset line 부분을 roi로 설정
    roi = edge_img[Offset: Offset+Gap, 0: Width]
    # HoughLinesP를 활용하여 모든 라인 검출, 각 라인의 시작점과 끝점을 return
    all_lines = cv2.HoughLinesP(roi, 1, math.pi/180, 30, 30, 10)

    # divide left, right lines
    if all_lines is None:
        return 0, 640

    left_lines, right_lines = divide_left_right(
        all_lines)              # 허프만으로 검출된 모든 line를 필터링 및 좌우로 분류

    # get center of lines
    # line이 그려진 이미지와 lpos,rpos 반환
    frame, lpos = get_line_pos(frame, left_lines, left=True)
    frame, rpos = get_line_pos(frame, right_lines, right=True)

    # draw lines
    #frame = draw_lines(frame, left_lines)
    #frame = draw_lines(frame, right_lines)
    #frame = cv2.line(frame, (230, 235), (410, 235),
    #                 (255, 255, 255), 2)   # 정면 흰색 선

    # draw rectangle
    #frame = draw_rectangle(frame, lpos, rpos, offset=Offset)

    # TODO4 함수 위치
    frame = print_corr(frame)

    # show image
    # cv2.imshow('calibration', frame)

    return lpos, rpos
    
def print_corr(frame):
    global cap
    text="Video Frame num : %d" % (cap.get(cv2.CAP_PROP_POS_FRAMES))
    org=(400,50)
    cv2.putText(frame,text,org,cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    cv2.imshonw('calibration', frame)
    cv2.setMouseCallback('calibration', onMouse)

    return frame

def onMouse(event, x, y, flags, param) :
    if event == cv2.EVENT_MOUSEMOVE:
        print('현재 마우스 좌표 : ', x, y)

def print_answer_rate():
    global line_temp
    #target=pd.read_csv('./Line_detection/target_final.csv',names=['lposl','lposr','rposl','rposr'], header=None)
    target=pd.read_csv('./Line_detection/pos.csv',encoding= 'utf-8')

    check_answerl = []
    check_answerr = []

    # for i in target.index:
    #     if abs(target["lposl"][i]-target["lposr"][i])<=2:
    #         check_answerl.append(1)
    #     elif target["lposl"][i]<=line_temp[i][0]  <=target["lposr"][i]:
    #         check_answerl.append(1)
    #     else:
    #         check_answerl.append(0)

    #     if abs(target["rposl"][i]-target["rposr"][i])<=2:
    #         check_answerr.append(1)
    #     elif target["rposl"][i]<=line_temp[i][1]  <=target["rposr"][i]:
    #         check_answerr.append(1)
    #     else:
    #         check_answerr.append(0)
    for i in target.index:
        if abs(target["new lposl"][i]-target["new lposr"][i])<=2:
            check_answerl.append(1)
        elif target["new lposl"][i]<=line_temp[i][0]  <=target["new lposr"][i]:
            check_answerl.append(1)
        else:
            check_answerl.append(0)

        if abs(target["new rposl"][i]-target["new rposr"][i])<=2:
            check_answerr.append(1)
        elif target["new rposl"][i]<=line_temp[i][1]  <=target["new rposr"][i]:
            check_answerr.append(1)
        else:
            check_answerr.append(0)    
    



    print("left correct answer rate : ", check_answerl.count(1)/len(check_answerl))
    print("right correct answer rate : ", check_answerr.count(1)/len(check_answerr))
    print("correct answer rate : {:2f}".format(((check_answerl.count(1)/len(check_answerl))+(check_answerr.count(1)/len(check_answerr)))/2))
    

    # target=pd.concat([target,pd.DataFrame(line_temp)],axis=1)
    # target=pd.concat([target,pd.DataFrame(check_answerl)],axis=1)
    # target=pd.concat([target,pd.DataFrame(check_answerr)],axis=1)
    # target.columns=["answer_lpos", "answer_rpos", "target_lposl", "target_lposr", "target_rposl","target_rposr","original_lcorrect","original_rcorrect"] 
    # target.to_csv('./Line_detection/check_correct1.csv', index=False)

def start():
    global cap
    global Width, Height
    global line_temp
    frame_idx=0

    while cap.isOpened():                                               # 실행할 수 있도록 수정
        run, frame = cap.read()
        
        if not run:
            print("end")
            break
        
        frame_idx+=1
        if frame_idx==30:
            lpos, rpos = process_image(frame)
            line_temp.append([lpos,rpos])
            # TODO 2 : 차선이 끊긴 구간은 위 아래의 차선을 판별하려 가상의 위치를 찍어야 하나? 맞다면 해당하는 함수
            
            # TODO 3 : 밝은 곳과 어두운 곳에 가변적으로 threshold 값이 바뀌어 적용되는 부분. 현재 기존 함수 내에 작성해야 할 것 같고 생각보다 허프만이 강력함
            
            frame_idx=0

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # key = cv2.waitKey(1) & 0xFF         # 키보드 입력 전 무한 대기
            # if key == ord('n'):
            #     continue
            # elif key == ord('q'):
            #     break
            # elif key == ord('r'):
            #     print(len(target_temp))
            
            #time.sleep(0.01)            # 천천히 출력하기 위해 추가 / 여기가 지연부분

    #line = pd.DataFrame(line_temp)
    #line.to_csv('./Line_detection/line.csv',header=False, index=False)
    



    print_answer_rate()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':          # 파이썬의 main 함수
    start()
