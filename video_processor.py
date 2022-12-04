#!/usr/bin/env python3

import cv2
import csv
import numpy as np

from configparser import ConfigParser

import sortedcollection
from sortedcollection import SortedCollection

class MyParser(ConfigParser):

    def as_dict(self):
        d = dict(self._sections)
        for k in d:
            d[k] = dict(self._defaults, **d[k])
            d[k].pop('__name__', None)
        return d


def add_transparent_image(background, foreground, x_offset=None, y_offset=None, shift_x=0, shift_y=0):
    bg_h, bg_w, bg_channels = background.shape
    fg_h, fg_w, fg_channels = foreground.shape

    assert bg_channels == 3, f'background image should have exactly 3 channels (RGB). found:{bg_channels}'
    assert fg_channels == 4, f'foreground image should have exactly 4 channels (RGBA). found:{fg_channels}'

    # center by default
    if x_offset is None: x_offset = (bg_w - fg_w) // 2
    if y_offset is None: y_offset = (bg_h - fg_h) // 2

    w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
    h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)

    if w < 1 or h < 1: return

    # clip foreground and background images to the overlapping regions
    bg_x = max(0, x_offset) + shift_x
    bg_y = max(0, y_offset) + shift_y
    fg_x = max(0, x_offset * -1)
    fg_y = max(0, y_offset * -1)

    h = min(h, bg_h - bg_y)
    w = min(w, bg_w - bg_x)

    #print('sizes:', w, h)

    foreground = foreground[fg_y:fg_y + h, fg_x:fg_x + w]
    background_subsection = background[bg_y:bg_y + h, bg_x:bg_x + w]


    # separate alpha and color channels from the foreground image
    foreground_colors = foreground[:, :, :3]
    alpha_channel = foreground[:, :, 3] / 255  # 0-255 => 0.0-1.0

    # construct an alpha_mask that matches the image shape
    alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))

    # combine the background with the overlay image weighted by alpha
    composite = background_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask

    # overwrite the section of the background image that has been updated
    background[bg_y:bg_y + h, bg_x:bg_x + w] = composite


def load_settings(settings_file):
    parser = MyParser()
    parser.read(settings_file)
    return parser.as_dict()['main']

def telemetry_time_to_seconds(telemetry_time):
    vals = telemetry_time.split(':')
    seconds = float(vals[-1])
    minutes = 0
    hours = 0
    if len(vals) > 1:
        minutes = int(vals[-2])
    if len(vals) > 2:
        hours = int(vals[-3])
    return hours * 60 * 60 + minutes * 60 + seconds

class Telemetry:
    def __init__(self, telemetry_file):
        with open(telemetry_file) as f:
            data = [{k: v for k, v in row.items()}
                 for row in csv.DictReader(f, skipinitialspace=True)]
            for element in data:
                #print(element['Time'], telemetry_time_to_seconds(element['Time']))
                element['time_seconds'] = telemetry_time_to_seconds(element['Time'])
        self.data = sortedcollection.SortedCollection(data, key=lambda x: x['time_seconds'])

    def get_row(self, video_time, sync_video_start, sync_video_finish, sync_telemetry_start, sync_telemetry_finish):

        if video_time < sync_video_start:
            return None
        if video_time >= sync_telemetry_finish:
            return None

        video_delta = sync_video_finish - sync_video_start
        telemetry_delta = sync_telemetry_finish - sync_telemetry_start

        passed_percent = (video_time - sync_video_start) / video_delta
        target_telemetry = sync_telemetry_start + passed_percent * telemetry_delta

        if target_telemetry < sync_telemetry_start:
            return None
        if target_telemetry >= sync_telemetry_finish:
            return None

        idx = self.data.find_ge_idx(target_telemetry)
        prev = max(0, idx - 1)

        curr_frame = self.data[idx]
        prev_frame = self.data[prev]

        curr_frame_time = curr_frame['time_seconds']
        prev_frame_time = prev_frame['time_seconds']

        time_left = (target_telemetry - prev_frame_time)
        time_delta = max(time_left, curr_frame_time - prev_frame_time)

        factor = time_left / time_delta

        result = {}
        for field_name in ('Alt(m)', 'GSpd(kmh)',):
            curr_value = float(curr_frame[field_name])
            prev_value = float(prev_frame[field_name])
            result[field_name] = prev_value + factor * (curr_value - prev_value)

        return result


def draw_text(img, txt, pos, color, size):
    font = cv2.FONT_HERSHEY_SIMPLEX
    pos = (int(pos[0] * img.shape[1]), int(pos[1] * img.shape[0]))

    img_txt = np.zeros((100, 600, 4), np.uint8)
    img_txt = cv2.putText(img_txt, txt, (0, 46), font, size[0], color, size[1], cv2.LINE_AA)

    img_txt = cv2.resize(img_txt, (0, 0), fx=0.5, fy=0.5)

    #cv2.imshow('txt', img_txt)

    #print(pos)

    #add_transparent_image(img, img_txt, pos[0], pos[1])
    add_transparent_image(img, img_txt, 0, 0, pos[0], pos[1]-30)

    return img

    #cv2.putText(image, "Test String", (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255, 255))

    #return cv2.putText(img, txt, pos, font, size[0], color, size[1], cv2.LINE_AA)


def main():

    settings = load_settings('settings.ini')
    telemetry = Telemetry(settings['telemetry_file'])

    sync_video_start = telemetry_time_to_seconds(settings['sync_video_start'])
    sync_video_finish = telemetry_time_to_seconds(settings['sync_video_finish'])
    sync_telemetry_start = telemetry_time_to_seconds(settings['sync_telemetry_start'])
    sync_telemetry_finish = telemetry_time_to_seconds(settings['sync_telemetry_finish'])

    cap = cv2.VideoCapture(settings['video_file'])

    map = cv2.imread('map.jpg')
    map = cv2.cvtColor(map, cv2.COLOR_RGB2RGBA)

    map[:, :, 3] = (128,)

    map = cv2.resize(map, (0, 0), fx=0.4, fy=0.4)




    if not cap.isOpened():
        print("Error opening video file")
        return
    
    # Read until video is completed
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break


        frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        #total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_time = frame_num / fps

        curr_telemetry = telemetry.get_row(
            video_time=frame_time,
            sync_video_start=sync_video_start,
            sync_video_finish=sync_video_finish,
            sync_telemetry_start=sync_telemetry_start,
            sync_telemetry_finish=sync_telemetry_finish,
        )



        out_str = f'{frame_time:.1f}'

        GREEN_COLOR = (155, 255, 155, 165)
        if curr_telemetry is not None:
            altitude = int(curr_telemetry['Alt(m)'])
            frame = draw_text(frame, f'm {altitude}', (0.75, 0.45), GREEN_COLOR, (1.5, 6))

            velocity = int(curr_telemetry['GSpd(kmh)'])
            frame = draw_text(frame, f'{velocity} km/h', (0.14, 0.45), GREEN_COLOR, (1.5, 6))

            #frame = draw_text(frame, f'Assist allied ground units', (0.4, 0.15), GREEN_COLOR, (1.3, 4))

            #out_str += f' Alt: {altitude}m  Vel: {velocity}kmh'

        add_transparent_image(frame, map, 0, 0, 80, 400)




        # font = cv2.FONT_HERSHEY_SIMPLEX
        # pos = (50, int(0.8*frame.shape[0]))
        # fontScale = 1
        # fontScale = 1
        # color = (255, 255, 255)
        # thickness = 2
        # frame = cv2.putText(frame, out_str, pos, font, fontScale, color, thickness, cv2.LINE_AA)

        #frame = draw_text(frame, out_str, (0.05, 0.9), (255, 255, 255, 255), (1, 2))

        cv2.imshow('Frame', frame)

        key_in = cv2.waitKey(10) & 0xFF

        if key_in == ord('q'):
            break

        if key_in == ord('d'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_num + fps * 5))
        if key_in == ord('a'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_num - fps * 5)))

        if key_in == ord('c'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_num + fps * 30))
        if key_in == ord('z'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_num - fps * 30)))
     
    # When everything done, release
    # the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
