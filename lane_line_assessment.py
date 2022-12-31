import cv2
import math
import constant
import numpy as np
import sys


class LaneLineAssessment:
    def __init__(self):
        
        self.process_flag = 0
        self.dashboard_detected_line_count = 0
        self.dashboard_detected_line = 0
        self.dashboard_detected_line_mat = []
        self.birdeye_verify_init = True
        self.transformation_matrix_init = True
        
        self.left_points = []
        self.right_points = []
        self.left_angle_data = []
        self.right_angle_data = []  # variable to store right lane line's angle detected using LSD
        
        self.birdeye_verify_count = 0
        self.birdeye_verification_src_reset_counter = 0
        self.birdeye_verify_function_call_count = 0
        self.biredeye_view_left_strict_lines = []
        self.biredeye_view_right_strict_lines = []
        self.birdview_left_proper = True
        self.birdview_right_proper = True
        self.source_temp_2_x = 0
        self.source_temp_3_x = 0
        self.scaled_source_temp_2_x = 0
        self.scaled_source_temp_3_x = 0
        self.stored_left_crop_coordinates_x1 = 0
        self.stored_left_crop_coordinates_x2 = 0
        self.stored_right_crop_coordinates_x1 = 0
        self.stored_right_crop_coordinates_x2 = 0
        
        self.transformation_matrix = []
        self.inv_transformation_matrix = []
        self.transformation_matrix_scaled = []
        self.inv_transformation_matrix_scaled = []
        
        self.transformation_source_point = np.zeros((4, 2), dtype="float32")
        self.transformation_source_point_scaled = np.zeros((4, 2), dtype="float32")
        self.transformation_destination_point = np.array([
            [constant.original_image_width, constant.original_image_height],
            [0, constant.original_image_height],
            [0, 0],
            [constant.original_image_width, 0]], dtype="float32")
        self.transformation_destination_point_scaled = np.array([
            [constant.scaled_image_width, constant.scaled_image_height],
            [0, constant.scaled_image_height],
            [0, 0],
            [constant.scaled_image_width, 0]], dtype="float32")
        
        self.compare_x_left = sys.maxsize
        self.compare_x_right = sys.maxsize * -1
        
        # create structure for processing
        self.horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (int(constant.x_pixel_shift * 0.4), 1))
        self.ls_birdeye_view = cv2.createLineSegmentDetector(_refine=cv2.LSD_REFINE_ADV, _scale=0.2, _sigma_scale=1.0,
                                                             _quant=2.0, _ang_th=10.0)
        self.lsd_matrix_verify_function_call_count = 0
        self.crop_top_line = 0
        
        self.lt_color = 0
        self.lt_outer_percentage = int(0)
        self.lt_inner_percentage = int(0)
        self.rt_color = 0
        self.rt_outer_percentage = int(0)
        self.rt_inner_percentage = int(0)

        # self.m_19_f_01 = "4"
        # self.m_19_f_07 = "000"
        # self.m_19_f_08 = "000"
        # self.m_19_f_09 = "000"
        # self.m_19_f_10 = "000"
        self.m_19_f_01 = ""
        self.m_19_f_07 = ""
        self.m_19_f_08 = ""
        self.m_19_f_09 = ""
        self.m_19_f_10 = ""
        self.m_19_overall_score = ""

    def lane_assessment_log(self):

        # if no line found set alarm flag as 4 and rest values as "000"
        if (self.lt_inner_percentage == 0) and (self.lt_outer_percentage == 0) and (
                self.rt_inner_percentage == 0) and \
                (self.rt_outer_percentage == 0):
            self.m_19_f_01 = "4"
            self.m_19_f_07 = "000"
            self.m_19_f_08 = "000"
            self.m_19_f_09 = "000"
            self.m_19_f_10 = "000"
            self.m_19_overall_score = "0"
        else:
            # if it is zero than set value as "000" generate log string based on calculated percentage if percentage
            # is less than 10 than add preceding zero to it
    
            # log id 19 field 07 | outer left lane result
            if self.lt_outer_percentage == 0:
                self.m_19_f_07 = "000"
            else:
                self.m_19_f_07 = str(self.lt_color) + str(
                    self.lt_outer_percentage) if self.lt_outer_percentage >= 10 else str(self.lt_color) + "0" + str(
                    self.lt_outer_percentage)
            # log id 19 field 08 | inner left lane result
            if self.lt_inner_percentage == 0:
                self.m_19_f_08 = "000"
            else:
                self.m_19_f_08 = str(self.lt_color) + str(
                    self.lt_inner_percentage) if self.lt_inner_percentage >= 10 else str(self.lt_color) + "0" + str(
                    self.lt_inner_percentage)
            if self.rt_inner_percentage == 0:
                self.m_19_f_09 = "000"
            else:
                self.m_19_f_09 = str(self.rt_color) + str(
                    self.rt_inner_percentage) if self.rt_inner_percentage >= 10 else str(self.rt_color) + "0" + str(
                    self.rt_inner_percentage)
            if self.rt_outer_percentage == 0:
                self.m_19_f_10 = "000"
            else:
                self.m_19_f_10 = str(self.rt_color) + str(
                    self.rt_outer_percentage) if self.rt_outer_percentage >= 10 else str(self.rt_color) + "0" + str(
                    self.rt_outer_percentage)
    
            # find min percentage of all four results also check if lane result is zero than make it 100 so that
            # lowest calculated percentage can be used to calculate alarm level
            min_t1 = 100 if self.lt_inner_percentage == 0 else self.lt_inner_percentage
            min_t2 = 100 if self.lt_outer_percentage == 0 else self.lt_outer_percentage
            min_t3 = 100 if self.rt_inner_percentage == 0 else self.rt_inner_percentage
            min_t4 = 100 if self.rt_outer_percentage == 0 else self.rt_outer_percentage
            assessment_alarm = min(min_t1, min_t2, min_t3, min_t4)
    
            self.m_19_overall_score = str(assessment_alarm)
    
            # init alarm level with 0 as if it is grater than 80 (>80)
            self.m_19_f_01 = "0"
    
            # now check for lower percentages and update alarm level
            # 60 <= assessment_alarm < 80
            if assessment_alarm < 80 and assessment_alarm >= 60:
                self.m_19_f_01 = "1"
            # 40 <= assessment_alarm < 60
            elif assessment_alarm < 60 and assessment_alarm >= 40:
                self.m_19_f_01 = "2"
            # 40 < assessment_alarm
            elif assessment_alarm < 40:
                self.m_19_f_01 = "3"
    
            min_t1 = 0
            min_t2 = 0
            min_t3 = 0
            min_t4 = 0

        self.lt_outer_percentage = int(0)
        self.lt_inner_percentage = int(0)
        self.rt_outer_percentage = int(0)
        self.rt_inner_percentage = int(0)
    
    def convert_to_grayscale(self, input_image):
        grayscale_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        return grayscale_image
    
    def horizontal_edge_detection(self, lane_assessment_input_gray):
        # apply Gaussian blur on src image
        gaussian_gray = cv2.GaussianBlur(lane_assessment_input_gray[
                                         constant.dd_crop_row_start:constant.dd_crop_row_start + constant.dd_crop_rows],
                                         (25, 25), 1, 1)
        # Run loop to crop and find best possible dashboard result
        col_number = constant.x_pixel_shift
        
        while col_number < (gaussian_gray.shape[1] - constant.x_pixel_shift):
            # crop image into a variable
            col_crop = gaussian_gray[:, int(col_number):int(col_number + constant.x_pixel_shift)]
            # cv2.imshow("col_crop " + str(col_number), col_crop)
            # cv2.waitKey(1)
            # apply adaptive threshold on image using below parameters
            bw = cv2.adaptiveThreshold(~col_crop, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
            
            # cv2.imshow("bw " + str(col_number), bw)
            # cv2.waitKey(1)
            # Create the images that will use to extract the horizontal and vertical lines
            horizontal = np.copy(bw)
            # Apply morphology operations
            horizontal = cv2.erode(horizontal, self.horizontalStructure, (-1, -1))
            # cv2.imshow("horizontal Erode " + str(col_number), horizontal)
            # cv2.waitKey(1)
            
            horizontal = cv2.dilate(horizontal, self.horizontalStructure, (-1, -1))
            # cv2.imshow("horizontal Dilate " + str(col_number), horizontal)
            # cv2.waitKey(1)
            
            # reduce horizontal to 1 column
            horizontal = cv2.reduce(horizontal, 1, cv2.REDUCE_MAX)
            # cv2.imshow("horizontal Reduce " + str(col_number), horizontal)
            # cv2.waitKey(1)
            
            # required variable to find best result
            start_crop = 0
            # run loop over the reduced 1 col result
            i = 0
            while i < horizontal.shape[0]:
                # if the value is greater than 0 and start_crop is zero than set value of start_crop to i
                if horizontal[i] > 0 & (start_crop == 0):
                    start_crop = i
                # if start_crop and end_crop are not zero set index1 variable to start_crop and break the loop
                if start_crop != 0:
                    self.dashboard_detected_line_mat.append(start_crop)
                    break
                i = i + 1
            col_number = col_number + (2 * constant.x_pixel_shift)
    
    def dashboard_detection(self, lane_assessment_input_gray):
        
        self.dashboard_detected_line_count += 1
        self.horizontal_edge_detection(lane_assessment_input_gray)
        
        # if dashboard_detected_line count is equal or greater than 25 set the dashboard line
        if (len(self.dashboard_detected_line_mat) >= 25) & (self.dashboard_detected_line_count >= 4):
            # sort values from maximum to minimum column wise
            temp_mat_01 = cv2.sort(np.asarray(self.dashboard_detected_line_mat),
                                   cv2.SORT_DESCENDING | cv2.SORT_EVERY_COLUMN)
            # set the maximum value as dashboard line here 480 needs to be added as in horizontal_edge_detection rows
            # from 480 to 720 processed to find dashboard
            self.dashboard_detected_line = 480 + temp_mat_01[0][0]
            self.dashboard_detected_line_mat.clear()
            self.dashboard_detected_line_count = 0
    
    def birdeye_view_verification(self, lane_assessment_input_color, lane_assessment_input_gray):
        gray_image = cv2.warpPerspective(lane_assessment_input_gray, self.transformation_matrix,
                                         (constant.original_image_width, constant.original_image_height),
                                         cv2.WARP_FILL_OUTLIERS + cv2.INTER_CUBIC)
        
        birdeye_view_lines = self.ls_birdeye_view.detect(gray_image)[0]
        strict_left_count = 0
        liberal_left_count = 0
        liberal_left_count_opp_side = 0
        
        strict_right_count = 0
        liberal_right_count = 0
        liberal_right_count_opp_side = 0
        original_image_width_half = 640
        for entry in birdeye_view_lines:
            x1 = int(entry[0][0])
            x2 = int(entry[0][2])
            y1 = int(entry[0][1])
            y2 = int(entry[0][3])
            temp_angle = (math.atan2(y1 - y2, x1 - x2) * 180) / math.pi
            temp_distance = math.sqrt((y1 - y2) * (y1 - y2) + (x1 - x2) * (x1 - x2))
            
            if temp_angle < 0:
                temp_angle = 360 + temp_angle
            
            if temp_distance < 70:
                pass
            elif ((90 <= temp_angle <= 100) or (270 <= temp_angle <= 280)) and (x1 < original_image_width_half) and \
                    (x2 < original_image_width_half):
                self.biredeye_view_left_strict_lines.append(entry)
                strict_left_count += 1
            
            elif ((80 <= temp_angle < 90) or (260 < temp_angle <= 270)) and (x1 >= original_image_width_half) and \
                    (x2 >= original_image_width_half):
                self.biredeye_view_right_strict_lines.append(entry)
                strict_right_count += 1
            elif ((100 < temp_angle <= 125) or (280 < temp_angle <= 305)) and (x1 < original_image_width_half) and \
                    (x2 < original_image_width_half):
                liberal_left_count += 1
            elif ((55 <= temp_angle < 80) or (235 <= temp_angle < 260)) and (x1 >= original_image_width_half) and \
                    (x2 >= original_image_width_half):
                liberal_right_count += 1
            elif ((85 <= temp_angle < 90) or (265 <= temp_angle < 270)) and (x1 < original_image_width_half) and \
                    (x2 < original_image_width_half):
                liberal_left_count_opp_side += 1
            elif ((90 < temp_angle <= 95) or (270 < temp_angle <= 275)) and (x1 >= original_image_width_half) and \
                    (x2 >= original_image_width_half):
                liberal_right_count_opp_side += 1
        
        if (strict_left_count + liberal_left_count >= 2) and \
                (strict_left_count > (liberal_left_count + liberal_left_count_opp_side)):
            self.birdview_left_proper = True
        else:
            self.birdeye_verify_count = 0
            self.birdeye_verification_src_reset_counter += 1
            if liberal_left_count < liberal_left_count_opp_side:
                self.transformation_source_point_scaled[2][0] = self.transformation_source_point_scaled[2][0] - (
                        13 / constant.input_scale)
                self.transformation_source_point[2][0] = self.transformation_source_point[2][0] - 13
                self.transformation_matrix = cv2.getPerspectiveTransform(self.transformation_source_point,
                                                                         self.transformation_destination_point)
                self.inv_transformation_matrix = cv2.getPerspectiveTransform(self.transformation_destination_point,
                                                                             self.transformation_source_point)
                self.transformation_matrix_scaled = cv2.getPerspectiveTransform(
                    self.transformation_source_point_scaled,
                    self.transformation_destination_point_scaled)
                self.inv_transformation_matrix_scaled = cv2.getPerspectiveTransform(
                    self.transformation_destination_point_scaled,
                    self.transformation_source_point_scaled)
            else:
                self.transformation_source_point_scaled[2][0] = self.transformation_source_point_scaled[2][0] + (
                        13 / constant.input_scale)
                self.transformation_source_point_scaled[3][0] = self.transformation_source_point_scaled[3][0] - (
                        13 / constant.input_scale)
                self.transformation_source_point[2][0] = self.transformation_source_point[2][0] + 13
                self.transformation_source_point[3][0] = self.transformation_source_point[3][0] - 13
                self.transformation_matrix = cv2.getPerspectiveTransform(self.transformation_source_point,
                                                                         self.transformation_destination_point)
                self.inv_transformation_matrix = cv2.getPerspectiveTransform(self.transformation_destination_point,
                                                                             self.transformation_source_point)
                self.transformation_matrix_scaled = cv2.getPerspectiveTransform(
                    self.transformation_source_point_scaled,
                    self.transformation_destination_point_scaled)
                self.inv_transformation_matrix_scaled = cv2.getPerspectiveTransform(
                    self.transformation_destination_point_scaled,
                    self.transformation_source_point_scaled)
        
        if (strict_right_count + liberal_right_count >= 2) and \
                (strict_right_count > (liberal_right_count + liberal_right_count_opp_side)):
            self.birdview_right_proper = True
        else:
            self.birdeye_verify_count = 0
            self.birdeye_verification_src_reset_counter += 1
            if liberal_right_count < liberal_right_count_opp_side:
                self.transformation_source_point_scaled[3][0] = self.transformation_source_point_scaled[3][0] + (
                        13 / constant.input_scale)
                self.transformation_source_point[3][0] = self.transformation_source_point[3][0] + 13
                self.transformation_matrix = cv2.getPerspectiveTransform(self.transformation_source_point,
                                                                         self.transformation_destination_point)
                self.inv_transformation_matrix = cv2.getPerspectiveTransform(self.transformation_destination_point,
                                                                             self.transformation_source_point)
                self.transformation_matrix_scaled = cv2.getPerspectiveTransform(
                    self.transformation_source_point_scaled,
                    self.transformation_destination_point_scaled)
                self.inv_transformation_matrix_scaled = cv2.getPerspectiveTransform(
                    self.transformation_destination_point_scaled,
                    self.transformation_source_point_scaled)
            else:
                self.transformation_source_point_scaled[2][0] = self.transformation_source_point_scaled[2][0] + (
                        13 / constant.input_scale)
                self.transformation_source_point_scaled[3][0] = self.transformation_source_point_scaled[3][0] - (
                        13 / constant.input_scale)
                self.transformation_source_point[2][0] = self.transformation_source_point[2][0] + 13
                self.transformation_source_point[3][0] = self.transformation_source_point[3][0] - 13
                self.transformation_matrix = cv2.getPerspectiveTransform(self.transformation_source_point,
                                                                         self.transformation_destination_point)
                self.inv_transformation_matrix = cv2.getPerspectiveTransform(self.transformation_destination_point,
                                                                             self.transformation_source_point)
                self.transformation_matrix_scaled = cv2.getPerspectiveTransform(
                    self.transformation_source_point_scaled,
                    self.transformation_destination_point_scaled)
                self.inv_transformation_matrix_scaled = cv2.getPerspectiveTransform(
                    self.transformation_destination_point_scaled,
                    self.transformation_source_point_scaled)
        
        if self.birdeye_verification_src_reset_counter >= 40:
            self.transformation_source_point[2][0] = self.source_temp_2_x
            self.transformation_source_point[3][0] = self.source_temp_3_x
            self.transformation_source_point_scaled[2][0] = self.scaled_source_temp_2_x
            self.transformation_source_point_scaled[3][0] = self.scaled_source_temp_3_x
            self.transformation_matrix = cv2.getPerspectiveTransform(self.transformation_source_point,
                                                                     self.transformation_destination_point)
            self.inv_transformation_matrix = cv2.getPerspectiveTransform(self.transformation_destination_point,
                                                                         self.transformation_source_point)
            self.transformation_matrix_scaled = cv2.getPerspectiveTransform(
                self.transformation_source_point_scaled,
                self.transformation_destination_point_scaled)
            self.inv_transformation_matrix_scaled = cv2.getPerspectiveTransform(
                self.transformation_destination_point_scaled,
                self.transformation_source_point_scaled)
            self.birdeye_verification_src_reset_counter = 0
        
        if abs(self.transformation_source_point_scaled[2][0] - self.transformation_source_point_scaled[3][0]) \
                <= (abs(self.scaled_source_temp_2_x - self.scaled_source_temp_3_x)) / 2:
            self.transformation_source_point[2][0] = self.source_temp_2_x
            self.transformation_source_point[3][0] = self.source_temp_3_x
            self.transformation_source_point_scaled[2][0] = self.scaled_source_temp_2_x
            self.transformation_source_point_scaled[3][0] = self.scaled_source_temp_3_x
            self.transformation_matrix = cv2.getPerspectiveTransform(self.transformation_source_point,
                                                                     self.transformation_destination_point)
            self.inv_transformation_matrix = cv2.getPerspectiveTransform(self.transformation_destination_point,
                                                                         self.transformation_source_point)
            self.transformation_matrix_scaled = cv2.getPerspectiveTransform(
                self.transformation_source_point_scaled,
                self.transformation_destination_point_scaled)
            self.inv_transformation_matrix_scaled = cv2.getPerspectiveTransform(
                self.transformation_destination_point_scaled,
                self.transformation_source_point_scaled)
            self.birdeye_verification_src_reset_counter = 0
        
        if self.birdview_left_proper and self.birdview_right_proper:
            self.birdeye_verify_count += 1
            print(self.birdeye_verify_count)
            if self.birdeye_verify_count >= 5:
                self.birdeye_verify_init = False
    
    def lsd_transformation_matrix_update(self, lane_assessment_input_gray):
        
        temp_mat_02 = lane_assessment_input_gray[self.crop_top_line:self.dashboard_detected_line,
                      constant.col_start_range:constant.col_end_range]
        
        lsd_src_update_lines = self.ls_birdeye_view.detect(temp_mat_02)[0]
        
        if len(temp_mat_02.shape) == 3:
            _, temp_mat_02_width, _ = temp_mat_02.shape
        else:
            _, temp_mat_02_width = temp_mat_02.shape
        
        temp_mat_02_width_half = temp_mat_02_width // 2
        if lsd_src_update_lines is not None:
            for entry in lsd_src_update_lines:
                x1 = int(entry[0][0])
                x2 = int(entry[0][2])
                y1 = int(entry[0][1])
                y2 = int(entry[0][3])

                temp_angle = int((math.atan2(y1 - y2, x1 - x2) * 180) / math.pi)
                temp_distance = math.sqrt((y1 - y2) * (y1 - y2) + (x1 - x2) * (x1 - x2))

                if temp_angle < 0:
                    temp_angle = 360 + temp_angle

                if temp_distance < 60:
                    pass
                elif ((110 <= temp_angle <= 160) or (290 <= temp_angle <= 340)) and (x1 < temp_mat_02_width_half) and \
                        (x2 < temp_mat_02_width_half) and (len(self.left_points) < 11):
                    self.left_points.append(entry)
                    if temp_angle > 180:
                        self.left_angle_data.append(temp_angle)
                    else:
                        self.left_angle_data.append(temp_angle + 180)
                    self.left_angle_data.append(temp_angle)
                elif ((20 <= temp_angle <= 70) or (200 <= temp_angle <= 250)) and (x1 >= temp_mat_02_width_half) and \
                        (x2 >= temp_mat_02_width_half) and (len(self.right_points) < 11):
                    self.right_points.append(entry)
                    if temp_angle > 180:
                        self.right_angle_data.append(temp_angle)
                    else:
                        self.right_angle_data.append(temp_angle + 180)
        
        len_r_a_d = len(self.right_angle_data)
        len_l_a_d = len(self.left_angle_data)
        x1_roi_left = 0
        y1_roi_left = 0
        x2_roi_left = 0
        y2_roi_left = 0
        x1_roi_right = 0
        y1_roi_right = 0
        x2_roi_right = 0
        y2_roi_right = 0
        
        if ((len_r_a_d >= 10) and (len_l_a_d >= 10)) or ((len_r_a_d >= 10) and
                                                         (len_l_a_d >= 5)) or ((len_r_a_d >= 5) and (len_l_a_d >= 10)):
           
            for left_index, left_point_entry in enumerate(self.left_points):
                if (left_point_entry[0][0] < self.compare_x_left) or (left_point_entry[0][2] < self.compare_x_left):
                    if left_point_entry[0][0] <= left_point_entry[0][2]:
                        self.compare_x_left = int(left_point_entry[0][0])
                    else:
                        self.compare_x_left = int(left_point_entry[0][2])
                    x1_roi_left = int(left_point_entry[0][0])
                    y1_roi_left = int(left_point_entry[0][1])
                    x2_roi_left = int(left_point_entry[0][2])
                    y2_roi_left = int(left_point_entry[0][3])
                    # angle_roi_left = self.left_angle_data[left_index]
            
            for right_index, right_point_entry in enumerate(self.right_points):
                if (right_point_entry[0][0] > self.compare_x_right) or (right_point_entry[0][2] > self.compare_x_right):
                    if right_point_entry[0][0] >= right_point_entry[0][2]:
                        self.compare_x_right = int(right_point_entry[0][0])
                    else:
                        self.compare_x_right = int(right_point_entry[0][2])
                    x1_roi_right = int(right_point_entry[0][0])
                    y1_roi_right = int(right_point_entry[0][1])
                    x2_roi_right = int(right_point_entry[0][2])
                    y2_roi_right = int(right_point_entry[0][3])
                    # angle_roi_right = self.right_angle_data[right_index]
            
            bottom_left_x = ((((x2_roi_left - constant.x_pixel_shift) - (x1_roi_left - constant.x_pixel_shift)) * (
                    self.dashboard_detected_line - (y1_roi_left + self.crop_top_line))) / (
                                     (y2_roi_left + self.crop_top_line) - (y1_roi_left + self.crop_top_line))) + (
                                    x1_roi_left - constant.x_pixel_shift)
            top_left_x = ((((x2_roi_left - constant.x_pixel_shift) - (x1_roi_left - constant.x_pixel_shift)) * (
                    self.crop_top_line - (y1_roi_left + self.crop_top_line))) / (
                                  (y2_roi_left + self.crop_top_line) - (y1_roi_left + self.crop_top_line))) + (
                                 x1_roi_left - constant.x_pixel_shift)
            bottom_right_x = ((((x2_roi_right + constant.x_pixel_shift) - (x1_roi_right + constant.x_pixel_shift)) * (
                    self.dashboard_detected_line - (y1_roi_right + self.crop_top_line))) / (
                                      (y2_roi_right + self.crop_top_line) - (y1_roi_right + self.crop_top_line))) + (
                                     x1_roi_right + 3 * constant.x_pixel_shift)
            top_right_x = ((((x2_roi_right + constant.x_pixel_shift) - (x1_roi_right + constant.x_pixel_shift)) * (
                    self.crop_top_line - (y1_roi_right + self.crop_top_line))) / (
                                   (y2_roi_right + self.crop_top_line) - (y1_roi_right + self.crop_top_line))) + (
                                  x1_roi_right + 3 * constant.x_pixel_shift)
            
            bottom_left_x = int(bottom_left_x)
            top_left_x = int(top_left_x)
            bottom_right_x = int(bottom_right_x)
            top_right_x = int(top_right_x)
            
            self.transformation_source_point[0] = [bottom_right_x, self.dashboard_detected_line]
            self.transformation_source_point[1] = [bottom_left_x, self.dashboard_detected_line]
            self.transformation_source_point[2] = [top_left_x + 80, self.crop_top_line]
            self.transformation_source_point[3] = [top_right_x - 80, self.crop_top_line]
            
            self.transformation_source_point_scaled[0] = self.transformation_source_point[0] / constant.input_scale
            self.transformation_source_point_scaled[1] = self.transformation_source_point[1] / constant.input_scale
            self.transformation_source_point_scaled[2] = self.transformation_source_point[2] / constant.input_scale
            self.transformation_source_point_scaled[3] = self.transformation_source_point[3] / constant.input_scale
            
            self.transformation_matrix = cv2.getPerspectiveTransform(self.transformation_source_point,
                                                                     self.transformation_destination_point)
            self.inv_transformation_matrix = cv2.getPerspectiveTransform(self.transformation_destination_point,
                                                                         self.transformation_source_point)
            self.transformation_matrix_scaled = cv2.getPerspectiveTransform(self.transformation_source_point_scaled,
                                                                            self.transformation_destination_point_scaled)
            self.inv_transformation_matrix_scaled = cv2.getPerspectiveTransform(
                self.transformation_destination_point_scaled,
                self.transformation_source_point_scaled)
            
            self.source_temp_2_x = int(self.transformation_source_point[2][0])
            self.source_temp_3_x = int(self.transformation_source_point[3][0])
            self.scaled_source_temp_2_x = int(self.transformation_source_point_scaled[2][0])
            self.scaled_source_temp_3_x = int(self.transformation_source_point_scaled[3][0])
            self.transformation_matrix_init = False
    
    def transform_image(self, input_image):
        
        if len(input_image.shape) == 3:
            max_height, max_width, _ = input_image.shape
        else:
            max_height, max_width = input_image.shape
        transformed_image = cv2.warpPerspective(input_image, self.transformation_matrix, (max_width, max_height),
                                                cv2.WARP_FILL_OUTLIERS + cv2.INTER_CUBIC)
        return transformed_image
    
    def remove_top_half_image(self, input_image):
        image_channels = len(input_image.shape)
        if image_channels == 3:
            return input_image[360:, :, :].copy()
        else:
            return input_image[360:, :].copy()
    
    def run_lsd(self, input_image):
        """
        Usage || is_lsd_proper, detected_lines = run_lsd(lane_assessment_input_gray)
        :param input_image:
        :return:
        """
        detected_lines = self.ls_birdeye_view.detect(input_image)[0]
        # if len(detected_lines) >= 1:
        if detected_lines is not None:
            return True, detected_lines
        else:
            return False, detected_lines
    
    def x_cord_cal(self, y_cord, x1, x2, y1, y2):
        if x1 == x2:
            return x1
        else:
            return int(float(((y_cord - y1) // round((y2 - y1) / (x2 - x1))) + x1))
    
    def filter_left_detected_lines(self, detected_lines, blank_left, left_line_pass_count):
        for entry in detected_lines:
            if min(entry[0]) > 0:
                x1 = int(entry[0][0])
                x2 = int(entry[0][2])
                y1 = int(entry[0][1])
                y2 = int(entry[0][3])
                temp_angle = (math.atan2(y1 - y2, x1 - x2) * 180) / math.pi
                temp_distance = math.sqrt((y1 - y2) * (y1 - y2) + (x1 - x2) * (x1 - x2))
                
                if temp_angle < 0:
                    temp_angle = 360 + temp_angle
                
                # if (360 / 3.75) < temp_distance < (360 * 1.744):
                if (temp_distance < (360 / 3.75)) or ( temp_distance > (360 * 1.744)):
                    pass
                elif (85 <= temp_angle <= 125) or (265 <= temp_angle <= 305):
                    y_cord_top, y_cord_bottom = 0, 720
                    x_cord_top = self.x_cord_cal(y_cord_top, x1, x2, y1, y2)
                    x_cord_bottom = self.x_cord_cal(y_cord_bottom, x1, x2, y1, y2)
                    cv2.line(blank_left, (x_cord_top, 0), (x_cord_bottom, 0), (255, 255, 255), 1)
                    left_line_pass_count += 1
        
        if left_line_pass_count != 0:
            return True
        else:
            return False
    
    def use_stored_left_coordinates(self, gray, color):
        left_lane_region_gray = 0,
        left_lane_region_color = 0
        if (self.stored_left_crop_coordinates_x1 + self.stored_left_crop_coordinates_x2) != 0:
            left_lane_region_gray = \
                gray[:, self.stored_left_crop_coordinates_x1:self.stored_left_crop_coordinates_x2].copy()
            left_lane_region_color = \
                color[:, self.stored_left_crop_coordinates_x1:self.stored_left_crop_coordinates_x2].copy()
            return True, left_lane_region_gray, left_lane_region_color
        else:
            return False, left_lane_region_gray, left_lane_region_color
    
    def find_crop_coordinate(self, blank_left):
        start_value, end_value = 0, 0
        intermediate_start_val, intermediate_end_val = 0, 0
        for count_forward in range(len(blank_left[0])):
            value = blank_left[0][count_forward]
            if value == 255:
                intermediate_start_val = count_forward
                break
        
        for count_backward in range(len(blank_left[0]) - 1, 0, -1):
            value = blank_left[0][count_backward]
            if value == 255:
                intermediate_end_val = count_backward
                break
        
        if (intermediate_start_val != 0) or (intermediate_end_val != 0):
            if intermediate_start_val - 40 <= 0:
                start_value = 0
            else:
                start_value = intermediate_start_val - 40
            
            if intermediate_end_val + 40 >= 640:
                end_value = 640
            else:
                end_value = intermediate_end_val + 40
            
            if (intermediate_end_val - intermediate_start_val) > (2.5 * 128):
                mean_val = (intermediate_start_val + intermediate_end_val) / 2
                seven_percentage_width = 1280 * 0.07
                if (mean_val - seven_percentage_width) <= 0:
                    start_value = 0
                else:
                    start_value = mean_val - seven_percentage_width
                
                if (start_value + (2 * seven_percentage_width)) >= 640:
                    end_value = 640
                else:
                    end_value = start_value + (2 * seven_percentage_width)
            return True, int(start_value), int(end_value)
        else:
            return False, start_value, end_value
    
    def crop_left_region(self, bview_input_image, bview_color):
        left_half_bview = bview_input_image[:, :bview_input_image.shape[1] // 2].copy()
        is_lsd_proper, detected_lines = self.run_lsd(left_half_bview)
        if is_lsd_proper:
            left_line_pass_count = 0
            blank_left = left_half_bview[:1, :].copy() * 0
            is_filtered_result_useful = self.filter_left_detected_lines(detected_lines, blank_left,
                                                                        left_line_pass_count)
            if is_filtered_result_useful:
                is_coordinates_found, crop_left_x1, crop_left_x2 = self.find_crop_coordinate(blank_left)
                if is_coordinates_found:
                    self.stored_left_crop_coordinates_x1 = crop_left_x1
                    self.stored_left_crop_coordinates_x2 = crop_left_x2
                    left_lane_region_gray = \
                        left_half_bview[:,
                        self.stored_left_crop_coordinates_x1:self.stored_left_crop_coordinates_x2].copy()
                    left_lane_region_color = \
                        bview_color[:, self.stored_left_crop_coordinates_x1:self.stored_left_crop_coordinates_x2].copy()
                else:
                    is_stored_coordinate_useful, left_lane_region_gray, left_lane_region_color = \
                        self.use_stored_left_coordinates(left_half_bview, bview_color)
                    if not is_stored_coordinate_useful:
                        return -24, 0, 0
            else:
                is_stored_coordinate_useful, left_lane_region_gray, left_lane_region_color = \
                    self.use_stored_left_coordinates(left_half_bview, bview_color)
                if not is_stored_coordinate_useful:
                    return -23, 0, 0
        else:
            is_stored_coordinate_useful, left_lane_region_gray, left_lane_region_color = \
                self.use_stored_left_coordinates(left_half_bview, bview_color)
            if not is_stored_coordinate_useful:
                return -22, 0, 0
        
        return 0, left_lane_region_gray, left_lane_region_color
    
    def custom_rgb2cmyk(self, image):
        cmyk_scale = 100
        b, g, r = cv2.split(image)
        c = 1 - r / 255.
        m = 1 - g / 255.
        y = 1 - b / 255.
        
        min_c = np.min(c)
        min_m = np.min(m)
        min_y = np.min(y)
        min_cmy = min(min_c, min_m, min_y)
        c = (c - min_cmy) / (1 - min_cmy)
        m = (m - min_cmy) / (1 - min_cmy)
        y = (y - min_cmy) / (1 - min_cmy)
        k = min_cmy
        c, m, y = np.asarray(c * cmyk_scale, dtype=np.uint8), np.asarray(m * cmyk_scale, dtype=np.uint8), np.asarray(
            y * cmyk_scale, dtype=np.uint8)
        k = np.asarray(k * cmyk_scale, dtype=np.uint8)
        return c, m, y, k
    
    def yellow_color_verification(self, new_img):
        
        la, a, b = cv2.split(cv2.cvtColor(new_img, cv2.COLOR_BGR2Lab))
        y_cr_cb_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2YCrCb)
        _, cr, cb = cv2.split(y_cr_cb_img)
        
        c, m, y, k = self.custom_rgb2cmyk(new_img)
        
        mask_cr = ((cr < 160) & (cr > 140)) | (cb < 100)
        mask_cmy = (c < 15) & (m < 15) & (y > 25)
        mask_lab = (la > 128) & (a > 123) & (a < 133) & (b > 150) & (b < 180)
        mask_cmy_lab = mask_cmy | mask_lab
        mask_cmy_lab_crcb = mask_cmy | mask_cr
        mask_cmy_img = np.uint8(mask_cmy * 255)
        
        mask_cmy_lab_img = np.uint8(mask_cmy_lab * 255)
        mask_cmy_lab_crcb_img = np.uint8(mask_cmy_lab_crcb * 255)
        pix_percent_cmy_lab_crcb = (np.sum(mask_cmy_lab_crcb_img) // 255) * 100 / (
                mask_cmy_img.shape[0] * mask_cmy_img.shape[1])
        
        # Decision part
        if 5 < pix_percent_cmy_lab_crcb < 70:
            return True
        else:
            return False
    
    def yellow_line_segmentation(self, new_img):
        # h, l, s = cv2.split(cv2.cvtColor(new_img, cv2.COLOR_BGR2HLS))
        # la, a, b = cv2.split(cv2.cvtColor(new_img, cv2.COLOR_BGR2Lab))
        
        # bird_eye_image = np.copy(new_img)
        # hsv_img = np.copy(cv2.cvtColor(new_img, cv2.COLOR_BGR2HSV))
        # hue, sat, val = cv2.split(hsv_img)
        gray_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
        # lab_img = np.copy(cv2.cvtColor(bird_eye_image, cv2.COLOR_BGR2Lab))
        
        # knn_part
        knn_train_data = np.array(
            [(0, 204, 225), (0, 102, 150), (0, 0, 255), (0, 0, 0), (150, 150, 150), (255, 255, 255)],
            dtype=np.float32)
        # knn_train_data = np.array([(21, 175, 198), (80, 142, 168),(90, 112, 131), (0, 0, 255), (0, 0, 0),
        #                           (150, 150, 150), (255, 255, 255)], dtype=np.float32)
        # knn_train_data = np.array([(115, 180, 240), (70, 144, 162), (0, 0, 255), (0, 0, 0), (150, 150, 150),
        #                           (255, 255, 255)], dtype=np.float32)
        knn_responses = np.array([[0], [1], [2], [3], [4], [5]])
        knn = cv2.ml.KNearest_create()
        knn.train(knn_train_data, cv2.ml.ROW_SAMPLE, knn_responses)
        knn_reshape_img = np.reshape(new_img, (-1, 3))
        knn_float_img = np.float32(knn_reshape_img)
        _, knn_results, knn_neighbours, _ = knn.findNearest(knn_float_img, 1)
        knn_labels_flattened = knn_neighbours.flatten()
        knn_labels_flattened = np.uint8(knn_labels_flattened)
        knn_sum_array = [np.sum(knn_labels_flattened == i) for i in range(np.size(knn_responses))]
        knn_sum_array = np.int32(knn_sum_array / (np.sum(knn_sum_array) * 0.01))
        knn_color_pallet = np.zeros_like(knn_train_data)
        knn_color_pallet[:] = knn_train_data[:]
        # knn_color = np.uint8(knn_color_pallet)
        # knn_output = knn_color[knn_labels_flattened]
        # knn_output_img = knn_output.reshape(new_img.shape)
        
        # If pixels corresponding to any of the yellow color intensities are found we determine mask
        # by setting corresponding pixel values to 255.
        if knn_sum_array[0] > 0 or knn_sum_array[1] > 0:
            knn_color_pallet = np.zeros_like(knn_train_data)
            knn_color_pallet[[0, 1]] = (255, 255, 255)
            knn_color = np.uint8(knn_color_pallet)
            intermediate_op = knn_color[knn_labels_flattened]
            intermediate_op_img = intermediate_op.reshape(new_img.shape)[:, :, 0]
        
        # Else we set pixels corresponding to black and grey color to 255 to create a mask
        # NOTE : Such condition occurs only when the is a very poor contrast of image
        else:
            knn_color_pallet = np.zeros_like(knn_train_data)
            knn_color_pallet[[4, 5]] = (255, 255, 255)
            knn_color = np.uint8(knn_color_pallet)
            intermediate_op = knn_color[knn_labels_flattened]
            intermediate_op_img = intermediate_op.reshape(new_img.shape)[:, :, 0]
        
        # segmenting image using mask generate with the above steps
        bitwise_and_img = cv2.bitwise_and(new_img, new_img, mask=intermediate_op_img)
        reshape_img = np.reshape(bitwise_and_img, (-1, 3))
        
        k_means_float_img = np.float32(reshape_img)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, k_means_label, k_means_center = cv2.kmeans(k_means_float_img, 3, None, criteria, 1, cv2.KMEANS_PP_CENTERS)
        
        k_means_center = np.uint8(k_means_center)
        k_means_labels_flattened = k_means_label.flatten()
        k_means_op = k_means_center[k_means_labels_flattened]
        k_means_op_img = k_means_op.reshape(new_img.shape)
        # ideal_hue = 26
        
        center_comp = k_means_center.reshape(-1, 1, 3)
        centers_gray = cv2.cvtColor(center_comp, cv2.COLOR_BGR2GRAY)
        # centers_hsv = cv2.cvtColor(center_comp, cv2.COLOR_BGR2HSV)
        # centers_hls = cv2.cvtColor(center_comp, cv2.COLOR_BGR2HLS)
        yellow_idx = np.argmax(centers_gray)
        # channel_value = centers_hsv[yellow_idx, 0, 0], centers_hsv[yellow_idx, 0, 1], centers_hsv[yellow_idx, 0, 2]
        # contrast_percent = centers_hsv[yellow_idx, 0, 1] / 2.55
        # hue_dev = np.std([centers_hsv[0, 0, 0], centers_hsv[1, 0, 0], centers_hsv[2, 0, 0]])
        # print("hue_dev",hue_dev)
        
        yellow_img = np.zeros_like(gray_img)
        re_label = k_means_labels_flattened.reshape(gray_img.shape)
        np.place(yellow_img, re_label == yellow_idx, 255)
        # yellow_pix_percent = (np.sum(yellow_img) // 255) * 100 / (yellow_img.shape[0] * yellow_img.shape[1])
        img_1 = np.zeros_like(gray_img)
        img_2 = np.zeros_like(gray_img)
        rmng_idx = list({0, 1, 2} - {yellow_idx})
        np.place(img_1, re_label == rmng_idx[0], 255)
        np.place(img_2, re_label == rmng_idx[1], 255)
        # cv2.imshow("op",np.hstack((yellow_img,img_1,img_2)))
        
        # *******************************************************************#
        # calculating vertical sum for cropping
        vertical_crop_img = np.zeros_like(gray_img)
        k_means_op_gray = cv2.cvtColor(k_means_op_img, cv2.COLOR_BGR2GRAY)
        np.place(vertical_crop_img, k_means_op_gray > 0, 255)
        bin_vertical_crop_img = vertical_crop_img // 255
        vertical_sum = np.sum(bin_vertical_crop_img, 0)
        
        # Taking 50 as threshold for extracting region of interest from k_means_op_img
        # width buffer so that yellow part don't miss
        vertical_x = np.where(vertical_sum >= 50)
        width_buffer = int(gray_img.shape[1] * 0.08)
        if len(vertical_x[0]) != 0:
            x_gray_min = np.min(vertical_x) - width_buffer
            x_gray_max = np.max(vertical_x) + width_buffer
            
            if x_gray_min < 0:
                x_gray_min = 0
        else:
            x_gray_min = 0
            x_gray_max = vertical_crop_img.shape[1] - 1
        
        # Cropped ROI, splitting h,s,v channel for segmentation.
        vertical_crop_img = new_img[:, x_gray_min:x_gray_max, :]
        gray_vertical_crop_img = cv2.cvtColor(vertical_crop_img, cv2.COLOR_BGR2GRAY)
        _, s_vertical_crop_img, _ = cv2.split(cv2.cvtColor(vertical_crop_img, cv2.COLOR_BGR2HSV))
        
        # segmenting vertical_crop_img using saturation channel
        v_crop_float_img = np.float32(s_vertical_crop_img.flatten())
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, v_crop_label, v_crop_center = cv2.kmeans(v_crop_float_img, 3, None, criteria, 1, cv2.KMEANS_PP_CENTERS)
        v_label_flattened = v_crop_label.flatten()
        v_center = np.uint8(v_crop_center)
        
        v_crop_reshape_label = np.reshape(v_label_flattened, (np.shape(gray_vertical_crop_img)))
        v_crop_seg_img = np.zeros_like(gray_vertical_crop_img)
        v_idx = np.argmax(v_center)
        # v_idx = np.where((np.argsort(v_center.flatten()))==1)
        np.place(v_crop_seg_img, v_crop_reshape_label == v_idx, v_center[v_idx])
        return v_crop_seg_img.copy(), vertical_crop_img
    
    def rotate_thresholded_image(self, input_image):
        pixels = cv2.findNonZero(input_image)
        y_la = []
        x_la = []
        if len(pixels) > 2:
            for i in pixels:
                y_la.append(i[0][1])
                x_la.append(i[0][0])
            
            poly = np.polyfit(y_la, x_la, 1)
            angle = math.atan(poly[0]) * 180 / math.pi
            rotation_matrix = cv2.getRotationMatrix2D((input_image.shape[1] // 2, input_image.shape[0] // 2), -angle, 1)
            rotated_image = cv2.warpAffine(input_image, rotation_matrix, (input_image.shape[1], input_image.shape[0]))
            return rotation_matrix, rotated_image
        else:
            return [], []
    
    def kmeans_2_cluster(self, input_image):
        compare_val = 130
        high_val_select = np.max(input_image)
        high_val = int(high_val_select)
        subtractor = (high_val - compare_val) // 3
        
        if subtractor < 20:
            subtractor = 20
        elif subtractor > 35:
            subtractor = 35
        
        break_loop_count = 0
        scaled_output_gray = input_image.copy() * 0
        
        while break_loop_count < 2:
            low_val = high_val - subtractor
            current_mask = cv2.inRange(input_image, low_val, high_val)
            current_mask = (current_mask // 255) * high_val
            high_val -= subtractor
            scaled_output_gray = scaled_output_gray | current_mask
            break_loop_count += 1
        
        data = np.float32(scaled_output_gray.flatten()).copy()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, colors = cv2.kmeans(data, 2, None, criteria, 1, cv2.KMEANS_PP_CENTERS | cv2.KMEANS_RANDOM_CENTERS)
        
        max_loc = np.argmax(colors, axis=0)
        threshold_image = np.reshape(np.uint8(labels == max_loc), input_image.shape)
        # threshold_image = np.uint8(np.reshape(labels, input_image.shape) * 255)
        return threshold_image
    
    def calculate_lane_percentage(self, input_th_image):
        temp_percentage_calc_2 = cv2.reduce(input_th_image, 1, cv2.REDUCE_MAX)
        
        row_start_pt = 0
        row_end_pt = temp_percentage_calc_2.shape[0]
        
        for count_forward in range((temp_percentage_calc_2.shape[0])):
            value = temp_percentage_calc_2[count_forward]
            if value == 255:
                row_start_pt = count_forward
                break
        
        for count_backward in range((temp_percentage_calc_2.shape[0]) - 1, 0, -1):
            value = temp_percentage_calc_2[count_backward]
            if value == 255:
                row_end_pt = count_backward
                break
        
        if row_start_pt == row_end_pt:
            return 0
        
        temp_percentage_calc_2 = input_th_image[row_start_pt: row_end_pt, :].copy()
        temp_percentage_calc = cv2.reduce(temp_percentage_calc_2, 0, cv2.REDUCE_MAX)
        start_pt = 0
        end_pt = temp_percentage_calc.shape[1]
        
        for count_forward in range((temp_percentage_calc.shape[1])):
            value = temp_percentage_calc[0][count_forward]
            if value == 255:
                start_pt = count_forward
                break
        for count_backward in range((temp_percentage_calc.shape[1]) - 1, 0, -1):
            value = temp_percentage_calc[0][count_backward]
            if value == 255:
                end_pt = count_backward
                break
        k = cv2.findNonZero(temp_percentage_calc_2[:, start_pt: end_pt])
        
        if k is None:
            return 0
        else:
            wd_ratio = int(k.shape[0] * 100.0 // (temp_percentage_calc_2.shape[0] * (end_pt - start_pt)))
            if wd_ratio == 100:
                wd_ratio = 99
            return wd_ratio
    
    def lt_yellow_lane_processing(self, lt_color_bview_crop_image):
        yellow_mat_segmented, yellow_mat_color = self.yellow_line_segmentation(lt_color_bview_crop_image)
        lt_gray_bview_crop_image = self.convert_to_grayscale(yellow_mat_color)
        temp_temp, lt_th_bview_crop_image = self.rotate_thresholded_image(yellow_mat_segmented)

        if (temp_temp == []):
            lt_th_bview_crop_image = yellow_mat_segmented.copy()
        else:
            temp1 = cv2.warpAffine(lt_gray_bview_crop_image, temp_temp,
                                   (lt_gray_bview_crop_image.shape[1], lt_gray_bview_crop_image.shape[0]))
            temp2 = cv2.warpAffine(yellow_mat_color, temp_temp,
                                   (yellow_mat_color.shape[1], yellow_mat_color.shape[0]))
            lt_gray_bview_crop_image = temp1.copy()
            yellow_mat_color = temp2.copy()
        
        lt_start_crop = 0
        lt_end_crop = 0
        lt_reduced_in_col = cv2.reduce(lt_th_bview_crop_image, 0, cv2.REDUCE_MAX)
        lt_cropped_regions_unfiltered_gray = []
        lt_cropped_regions_color = []
        lt_all_cropped_region_width = []
        lt_cropped_regions = []
        for i, value in enumerate(lt_reduced_in_col[0]):
            if value > 0 and lt_start_crop == 0:
                lt_start_crop = i
            elif (value == 0 and lt_start_crop != 0):
                lt_end_crop = i
            
            if lt_start_crop != 0 and lt_end_crop != 0 and (lt_end_crop - lt_start_crop >= 10):
                temp_mat_2 = lt_gray_bview_crop_image[:, lt_start_crop:lt_end_crop]
                temp_color = yellow_mat_color[:, lt_start_crop:lt_end_crop]
                lt_th_idx = cv2.findNonZero(temp_mat_2)
                
                if len(lt_th_idx) > (temp_mat_2.size * 0.15):
                    lt_cropped_regions_unfiltered_gray.append(temp_mat_2)
                    lt_cropped_regions_color.append(temp_color)
                    # cv2.imshow("lt_cropped_regions_color_" + str(len(lt_cropped_regions_color)), lt_cropped_regions_color[len(lt_cropped_regions_color) - 1])
                    # cv2.waitKey(1)
                    lt_all_cropped_region_width.append(temp_color.shape[1])
                    lt_start_crop = 0
                    lt_end_crop = 0
        # cv2.waitKey(0)
        """
        HERE ONE LOOP IS WRITTEN WHICH IS NOT REQUIRED TO BE IMPLEMENTED. BELOW IS SMALL REPLACEMENT OF THAT LOOP.
        """
        if len(lt_cropped_regions_color) > 0:
            lt_cropped_regions = lt_cropped_regions_unfiltered_gray.copy()
        else:
            return 32
        
        if len(lt_cropped_regions) == 2:
            temp_mat_1 = self.kmeans_2_cluster(lt_cropped_regions[0])
            self.lt_outer_percentage = self.calculate_lane_percentage(temp_mat_1)
            
            temp_mat_1 = self.kmeans_2_cluster(lt_cropped_regions[1])
            self.lt_inner_percentage = self.calculate_lane_percentage(temp_mat_1)
        elif len(lt_cropped_regions) == 1:
            temp_mat_1 = self.kmeans_2_cluster(lt_cropped_regions[0])
            self.lt_inner_percentage = self.calculate_lane_percentage(temp_mat_1)
        elif len(lt_cropped_regions) == 0:
            return 31
        else:
            """
            This logic decides that which 2 cropped regions have higher width.
            And they will passed for the assessment.
            Replacing that logic with simplified logic. and also added one logic above to make it easier.
            """
            ultimate = 1
            penultimate = 0
            if lt_all_cropped_region_width[0] >= lt_all_cropped_region_width[1]:
                ultimate = 0
                penultimate = 1
            
            ultimate_width = lt_all_cropped_region_width[ultimate]
            penultimate_width = lt_all_cropped_region_width[penultimate]
            
            for i in range(2, len(lt_all_cropped_region_width)):
                if lt_all_cropped_region_width[i] > ultimate_width:
                    penultimate = ultimate
                    penultimate_width = ultimate_width
                    ultimate = i
                    ultimate_width = lt_all_cropped_region_width[i]
                else:
                    if lt_all_cropped_region_width[i] > penultimate_width:
                        penultimate = i
                        penultimate_width = lt_all_cropped_region_width[i]
            
            temp_mat_1 = self.kmeans_2_cluster(lt_cropped_regions[ultimate])
            temp_mat_2 = self.kmeans_2_cluster(lt_cropped_regions[penultimate])
            
            if ultimate < penultimate:
                self.lt_outer_percentage = self.calculate_lane_percentage(temp_mat_1)
                self.lt_inner_percentage = self.calculate_lane_percentage(temp_mat_2)
            else:
                self.lt_outer_percentage = self.calculate_lane_percentage(temp_mat_2)
                self.lt_inner_percentage = self.calculate_lane_percentage(temp_mat_1)
        return 0
    
    def find_image_assessable(self, img):
        # param img : RGB or BGR image
        
        # This function will the identify images having high intensity variation.
        
        # HSV scale is perfect for finding color illumination in saturation channel.
        # First converting BGR image to HSV channel
        ycbcr_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        Y = cv2.extractChannel(ycbcr_image, 0)
        BGR = cv2.cvtColor(Y, cv2.COLOR_GRAY2BGR)
       
        hsv_img = cv2.cvtColor(BGR, cv2.COLOR_BGR2HSV)
        
        # Splitting the hsv_img in Hue(H), saturation(S) and value(V) region
        h, s, v = cv2.split(hsv_img)
        
        # If we check S channel of the image you will observe following output.
        # i) Low variation images :- S channel output will be dark.(its range wwould be around 0 - 25), Using human
        # Eye its hard to detect this much low variation.
        # ii) Medium variation :- S- channel output will change little bit, but its deviation from mean value will be
        # bounded in 10%.
        # iii) High variation :- S-channel output intensity will show clear variation, Std deviation(>20% mean)
        # and pixel intensity range would cross 100 mark in gray scale(0-255).
        
        # Apply k-means clustering on S-channel pixel intensity
        s_float = np.float32(s)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        # Set flags (Just to avoid line break in the code)
        flags = cv2.KMEANS_RANDOM_CENTERS
        s_comp, s_labels, s_centre = cv2.kmeans(s_float.flatten(), 3, None, criteria, 10, flags)
        # The reason for applying k_means is two folded,
        # i) Selecting proper threshold from histogram, mean and std deviation is very complex.
        # ii) Manually separating pixel value in different group is also complicated.
        # Using k_means clustering we can easily separate pixel values as per their intensity value
        # on different bins.
        
        # Below we are calculating high intensity value group. Number of pixel in each group.
        max_s_centre = max(s_centre)
        cluster_vol = [len(np.where(s_labels == i)[-1]) for i in range(len(s_centre))]
        
        # Calculating mean and std deviation from three centre value, reason for doing is
        # If the image is good centre value will lie close to each other and their deviation will low.
        # [print(s_centre[i] , cluster_vol[i]) for i in range(len(s_centre))]
        s_mean, s_std = np.mean(s_centre), np.std(s_centre)
        # print("mean = {}, std = {}".format(sk_mean, sk_std))
        
        # From observation we find that, if image satisfied below condition then definitely
        #                                                                            it has intensity variation
        # i) Centre value of the 3rd (high intensity) group of pixels crosses 80 mark in (0-255) range ,
        # ii) Std deviation more than 10% of the mean and
        # iii) 10% of the total pixel(S-Channel) lies in the 3rd(high intensity) group.
        
        percentage_p_last_bin = (cluster_vol[np.argmax(s_centre)] / len(s_labels)) * 100
        # print("percentage pixel in last_bin", percentage_p_last_bin)
        
        if (max_s_centre > 80 and s_std > (s_mean * 0.15) and percentage_p_last_bin > 7):
            return False
        else:
            return True
    
    def apply_mask(self, matrix, mask, fill_value):
        masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
        return masked.filled()
    
    def apply_threshold(self, matrix, low_value, high_value):
        low_mask = matrix < low_value
        matrix = self.apply_mask(matrix, low_mask, low_value)
        high_mask = matrix > high_value
        matrix = self.apply_mask(matrix, high_mask, high_value)
        
        return matrix
    
    def intensity_balancing_algorithm(self, input_image, percent):
        assert input_image.shape[2] == 3
        assert 0 < percent < 100
        half_percent = percent / 200.0
        hsv_img = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
        temp_split = cv2.split(hsv_img)
        out_channels = []
        out_channels.append(temp_split[0])
        out_channels.append(temp_split[1])
        
        height, width = temp_split[2].shape
        vec_size = width * height
        flat = temp_split[2].reshape(vec_size).copy()
        flat = np.sort(flat)
        
        n_cols = flat.shape[0]
        
        low_val = flat[math.floor(n_cols * half_percent)]
        high_val = flat[math.ceil(n_cols * (1.0 - half_percent))]
        thresholded = self.apply_threshold(temp_split[2], low_val, high_val)
        normalized = cv2.normalize(thresholded, thresholded.copy(), 0, 255, cv2.NORM_MINMAX)
        out_channels.append(normalized)
        out1 = cv2.merge(out_channels)
        out = cv2.cvtColor(out1, cv2.COLOR_HSV2BGR)
        # cv2.imshow("input", input_image)
        # cv2.imshow("output", out)
        # cv2.waitKey(0)
        return out
    
    def line_color_verification(self, new_img):
        gray_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
        reshape_img = np.reshape(gray_img, (-1, 1)).copy()
        data_gray = np.float32(reshape_img).copy()
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, label_gray, center_gray = \
            cv2.kmeans(data_gray, 3, None, criteria, 1, cv2.KMEANS_PP_CENTERS)
        
        max_gray_value = np.uint8(np.max(center_gray))
        
        if max_gray_value > 195:
            flag = 2
        else:
            is_image_assessable = self.find_image_assessable(new_img)
            if is_image_assessable:
                new_image_intensity = self.intensity_balancing_algorithm(new_img, 1)
                new_gray_intensity = cv2.cvtColor(new_image_intensity, cv2.COLOR_BGR2GRAY)
                reshape_img1 = np.reshape(new_gray_intensity, (-1, 1)).copy()
                data_gray1 = np.float32(reshape_img1).copy()
                _, label_gray1, center_gray1 = \
                    cv2.kmeans(data_gray1, 3, None, criteria, 1, cv2.KMEANS_PP_CENTERS)
                max_gray_value1 = np.uint8(np.max(center_gray1))
                
                if max_gray_value1 > 195:
                    flag = 2
                else:
                    flag = 0
            else:
                flag = 0
        return flag
    
    def white_lane_checker(self, input_image):
        temp_reduce = cv2.reduce(input_image / 255, 1, cv2.REDUCE_SUM)
        temp_reduce_1 = ((np.uint8(7 < temp_reduce)) & (np.uint8(temp_reduce < 40)))
        temp_reduce = input_image.copy() * 0
        idx = cv2.findNonZero(temp_reduce_1)
        if idx is None:
            return -1, []
        idx_ch1 = cv2.extractChannel(idx, 1)
        for i in idx_ch1:
            temp_reduce[i] = input_image[i].copy()
        
        temp_row_processed = temp_reduce.copy()
        temp_reduce_1 = cv2.reduce(temp_reduce / 255, 0, cv2.REDUCE_SUM)
        temp_reduce = (np.uint8(temp_reduce_1 > (len(idx_ch1) * 0.2)))
        idx = cv2.findNonZero(temp_reduce)
        
        if idx is None:
            return -1, []
        idx_ch1 = cv2.extractChannel(idx, 1)
        for i in idx_ch1:
            temp_reduce_1[i] = temp_row_processed[i].copy()
        
        temp_reduce = cv2.reduce(temp_reduce_1, 0, cv2.REDUCE_MAX)
        cropped_regions = []
        start_crop = 0
        end_crop = 0
        for i, value in enumerate(temp_reduce[0]):
            if value > 0 and start_crop == 0:
                start_crop = i
            elif value == 0 and start_crop != 0:
                end_crop = i
            if start_crop != 0 and end_crop != 0 and ((end_crop - start_crop) >= 10) and (
                    (end_crop - start_crop) <= 40):
                idx = cv2.findNonZero(input_image[:, start_crop:end_crop])
                if len(idx) > (input_image.shape[0] * (end_crop - start_crop) * 0.15):
                    cropped_regions.append(input_image[:, start_crop:end_crop])
                    start_crop = 0
                    end_crop = 0
        cropped_white_lane_region_1 = 0
        find_lane_mat = sys.maxsize * -1
        if len(cropped_regions) == 1:
            cropped_white_lane_region_1 = cropped_regions[0].copy()
        elif len(cropped_regions) == 0:
            return -1, []
        elif len(cropped_regions) > 2:
            for i, value in enumerate(cropped_regions):
                idx = cv2.findNonZero(cropped_regions[i])
                if find_lane_mat < len(idx):
                    find_lane_mat = i
            cropped_white_lane_region_1 = cropped_regions[find_lane_mat].copy()
        else:
            return -1, []
        
        return 0, cropped_white_lane_region_1
    
    def kmeans_clustering(self, input_image, lane_id):
        
        hsv_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
        s_ch_hsv_image = cv2.extractChannel(hsv_image, 1)
        s_ch_hsv_image = np.float32(s_ch_hsv_image).copy()
        data = s_ch_hsv_image.flatten()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, colors = \
            cv2.kmeans(data, 3, None, criteria, 1, cv2.KMEANS_RANDOM_CENTERS)
        _, max_val, _, max_loc = cv2.minMaxLoc(colors.flatten())
        # colors = cv2.reduce(colors, 1, cv2.REDUCE_MAX)
        mean_mat, stddev_mat = cv2.meanStdDev(colors)
        idx_max_number = cv2.countNonZero(np.uint8(labels == max_loc[0]))
        
        if (max_val > 80) and (stddev_mat[0][0] > (mean_mat[0][0] * 0.10)) and \
                ((idx_max_number / (input_image.shape[0] * input_image.shape[1])) > 0.10):
            # print("False")
            pass
        else:
            gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
            data = gray_image.flatten().copy()
            data = np.float32(data)
            _, labels, colors = \
                cv2.kmeans(data, 3, None, criteria, 1, cv2.KMEANS_RANDOM_CENTERS)
            _, max_val, _, max_loc = cv2.minMaxLoc(colors.flatten())
            threshold_image = np.reshape(np.uint8(labels == max_loc[0]) * 255, gray_image.shape)
            
            enable_white_flag = 0
            
            if enable_white_flag == 0:
                temp_temp, rotated_threshold_image = self.rotate_thresholded_image(threshold_image)
                if (temp_temp == []):
                    rotated_threshold_image = threshold_image.copy()

                reduced_threshold_image = cv2.reduce(rotated_threshold_image, 0, cv2.REDUCE_MAX)
                
                """
                here one loop is written which has no effect on current code. Skipping that loop for now.
                """
            
            white_checker, cropped_white_lane_region_1 = self.white_lane_checker(rotated_threshold_image)
            if white_checker != 0:
                pass
            else:
                if lane_id == "left":
                    self.lt_inner_percentage = self.calculate_lane_percentage(cropped_white_lane_region_1)
                else:
                    self.rt_inner_percentage = self.calculate_lane_percentage(cropped_white_lane_region_1)
    
    def lt_white_lane_processing(self, lt_color_bview_crop_image):
        lane_left_id = "left"
        self.kmeans_clustering(lt_color_bview_crop_image, lane_left_id)
        return 0
    
    def filter_right_detected_lines(self, detected_lines, blank_right, right_line_pass_count):
        for entry in detected_lines:
            if min(entry[0]) > 0:
                x1 = int(entry[0][0])
                x2 = int(entry[0][2])
                y1 = int(entry[0][1])
                y2 = int(entry[0][3])
                temp_angle = (math.atan2(y1 - y2, x1 - x2) * 180) / math.pi
                temp_distance = math.sqrt((y1 - y2) * (y1 - y2) + (x1 - x2) * (x1 - x2))
                
                if temp_angle < 0:
                    temp_angle = 360 + temp_angle
                
                # if (360 / 3.75) < temp_distance < (360 * 1.744):
                if (temp_distance < (360 / 3.75)) or (temp_distance > (360 * 1.744)):
                    pass
                elif (55 <= temp_angle <= 95) or (235 <= temp_angle <= 275):
                    y_cord_top, y_cord_bottom = 0, 720
                    x_cord_top = self.x_cord_cal(y_cord_top, x1, x2, y1, y2)
                    x_cord_bottom = self.x_cord_cal(y_cord_bottom, x1, x2, y1, y2)
                    cv2.line(blank_right, (x_cord_top, 0), (x_cord_bottom, 0), (255, 255, 255), 1)
                    right_line_pass_count += 1
        
        if right_line_pass_count != 0:
            return True
        else:
            return False
    
    def use_stored_right_coordinates(self, gray, color):
        right_lane_region_gray = 0
        right_lane_region_color = 0
        if (self.stored_right_crop_coordinates_x1 + self.stored_right_crop_coordinates_x2) != 0:
            right_lane_region_gray = \
                gray[:, gray.shape[1] // 2 + self.stored_right_crop_coordinates_x1:
                        gray.shape[1] // 2 + self.stored_right_crop_coordinates_x2].copy()
            right_lane_region_color = \
                color[:, color.shape[1] // 2 + self.stored_right_crop_coordinates_x1:
                         color.shape[1] // 2 + self.stored_right_crop_coordinates_x2].copy()
            return True, right_lane_region_gray, right_lane_region_color
        else:
            return False, right_lane_region_gray, right_lane_region_color
    
    def crop_right_region(self, bview_input_image, bview_color):

        right_half_bview = bview_input_image[:, bview_input_image.shape[1] // 2:].copy()
        is_lsd_proper, detected_lines = self.run_lsd(right_half_bview)
        if is_lsd_proper:
            right_line_pass_count = 0
            blank_right = right_half_bview[:1, :].copy() * 0
            is_filtered_result_useful = self.filter_right_detected_lines(detected_lines, blank_right,
                                                                         right_line_pass_count)
            if is_filtered_result_useful:
                is_coordinates_found, crop_right_x1, crop_right_x2 = self.find_crop_coordinate(blank_right)
                if is_coordinates_found:
                    self.stored_right_crop_coordinates_x1 = crop_right_x1
                    self.stored_right_crop_coordinates_x2 = crop_right_x2
                    right_lane_region_gray = \
                        bview_input_image[:, bview_input_image.shape[1] // 2 + self.stored_right_crop_coordinates_x1:
                                             bview_input_image.shape[1] // 2 + self.stored_right_crop_coordinates_x2].copy()
                    right_lane_region_color = \
                        bview_color[:, bview_input_image.shape[1] // 2 + self.stored_right_crop_coordinates_x1:
                                       bview_input_image.shape[1] // 2 + self.stored_right_crop_coordinates_x2].copy()
                else:
                    is_stored_coordinate_useful, right_lane_region_gray, right_lane_region_color = \
                        self.use_stored_right_coordinates(bview_input_image, bview_color)
                    if not is_stored_coordinate_useful:
                        return -27, 0, 0
            else:
                is_stored_coordinate_useful, right_lane_region_gray, right_lane_region_color = \
                    self.use_stored_right_coordinates(bview_input_image, bview_color)
                if not is_stored_coordinate_useful:
                    return -26, 0, 0
        else:
            is_stored_coordinate_useful, right_lane_region_gray, right_lane_region_color = \
                self.use_stored_right_coordinates(bview_input_image, bview_color)
            if not is_stored_coordinate_useful:
                return -25, 0, 0
        
        return 0, right_lane_region_gray, right_lane_region_color

    def rt_yellow_lane_processing(self, rt_color_bview_crop_image):
        yellow_mat_segmented, yellow_mat_color = self.yellow_line_segmentation(rt_color_bview_crop_image)
        rt_gray_bview_crop_image = self.convert_to_grayscale(yellow_mat_color)
        temp_temp, rt_th_bview_crop_image = self.rotate_thresholded_image(yellow_mat_segmented)
        if (temp_temp == []):
            rt_th_bview_crop_image = yellow_mat_segmented.copy()
        else:
            temp1 = cv2.warpAffine(rt_gray_bview_crop_image, temp_temp,
                                   (rt_gray_bview_crop_image.shape[1], rt_gray_bview_crop_image.shape[0]))
            temp2 = cv2.warpAffine(yellow_mat_color, temp_temp,
                                   (yellow_mat_color.shape[1], yellow_mat_color.shape[0]))
            rt_gray_bview_crop_image = temp1.copy()
            yellow_mat_color = temp2.copy()
    
        rt_start_crop = 0
        rt_end_crop = 0
        rt_reduced_in_col = cv2.reduce(rt_th_bview_crop_image, 0, cv2.REDUCE_MAX)
        rt_cropped_regions_unfiltered_gray = []
        rt_cropped_regions_color = []
        rt_all_cropped_region_width = []
        rt_cropped_regions = []
        for i, value in enumerate(rt_reduced_in_col[0]):
            if value > 0 and rt_start_crop == 0:
                rt_start_crop = i
            elif value == 0 and rt_start_crop != 0:
                rt_end_crop = i
        
            if rt_start_crop != 0 and rt_end_crop != 0 and (rt_end_crop - rt_start_crop >= 10):
                temp_mat_2 = rt_gray_bview_crop_image[:, rt_start_crop:rt_end_crop]
                temp_color = yellow_mat_color[:, rt_start_crop:rt_end_crop]
                rt_th_idx = cv2.findNonZero(temp_mat_2)
            
                if len(rt_th_idx) > (temp_mat_2.size * 0.15):
                    rt_cropped_regions_unfiltered_gray.append(temp_mat_2)
                    rt_cropped_regions_color.append(temp_color)
                    rt_all_cropped_region_width.append(temp_color.shape[1])
                    rt_start_crop = 0
                    rt_end_crop = 0
    
        """
           HERE ONE LOOP IS WRITTEN WHICH IS NOT REQUIRED TO BE IMPLEMENTED. BELOW IS SMALL REPLACEMENT OF THAT LOOP.
        """
        if len(rt_cropped_regions_color) > 0:
            rt_cropped_regions = rt_cropped_regions_unfiltered_gray.copy()
        else:
            return 32
    
        if len(rt_cropped_regions) == 2:
            temp_mat_1 = self.kmeans_2_cluster(rt_cropped_regions[0])
            self.rt_inner_percentage = self.calculate_lane_percentage(temp_mat_1)
        
            temp_mat_1 = self.kmeans_2_cluster(rt_cropped_regions[1])
            self.rt_outer_percentage = self.calculate_lane_percentage(temp_mat_1)
        elif len(rt_cropped_regions) == 1:
            temp_mat_1 = self.kmeans_2_cluster(rt_cropped_regions[0])
            self.rt_inner_percentage = self.calculate_lane_percentage(temp_mat_1)
        elif len(rt_cropped_regions) == 0:
            return 32
        else:
            """
            This logic decides that which 2 cropped regions have higher width.
            And they will passed for the assessment.
            Replacing that logic with simplified logic. and also added one logic above to make it easier.
            """
            ultimate = 1
            penultimate = 0
            if rt_all_cropped_region_width[0] >= rt_all_cropped_region_width[1]:
                ultimate = 0
                penultimate = 1
        
            ultimate_width = rt_all_cropped_region_width[ultimate]
            penultimate_width = rt_all_cropped_region_width[penultimate]
        
            for i in range(2, len(rt_all_cropped_region_width)):
                if rt_all_cropped_region_width[i] > ultimate_width:
                    penultimate = ultimate
                    penultimate_width = ultimate_width
                    ultimate = i
                    ultimate_width = rt_all_cropped_region_width[i]
                else:
                    if rt_all_cropped_region_width[i] > penultimate_width:
                        penultimate = i
                        penultimate_width = rt_all_cropped_region_width[i]
        
            temp_mat_1 = self.kmeans_2_cluster(rt_cropped_regions[ultimate])
            temp_mat_2 = self.kmeans_2_cluster(rt_cropped_regions[penultimate])
        
            if ultimate < penultimate:
                self.rt_inner_percentage = self.calculate_lane_percentage(temp_mat_1)
                self.rt_outer_percentage = self.calculate_lane_percentage(temp_mat_2)
            else:
                self.rt_inner_percentage = self.calculate_lane_percentage(temp_mat_2)
                self.rt_outer_percentage = self.calculate_lane_percentage(temp_mat_1)
    
        return 0

    def rt_white_lane_processing(self, rt_color_bview_crop_image):
        lane_right_id = "right"
        self.kmeans_clustering(rt_color_bview_crop_image, lane_right_id)
        return 0
    
    def pipeline(self, input_image):
        bview_color_full = self.transform_image(input_image)
        
        # cv2.imshow("bview_color_full", bview_color_full)
        # cv2.waitKey(1)
        
        bview_gray_full = self.convert_to_grayscale(bview_color_full)
        bview_color = self.remove_top_half_image(bview_color_full)
        bview_gray = self.remove_top_half_image(bview_gray_full)
        
        is_left_lane_cropped, left_lane_region_gray, left_lane_region_color = \
            self.crop_left_region(bview_gray, bview_color)
        
        if is_left_lane_cropped == 0:
            # cv2.imshow("left_lane_region_color", left_lane_region_color)
            # cv2.waitKey(1)
            is_left_image_yellow = self.yellow_color_verification(left_lane_region_color)
            if is_left_image_yellow:
                yellow_temp_result = self.lt_yellow_lane_processing(left_lane_region_color)
                self.lt_color = 1
            else:
                is_left_image_white = self.line_color_verification(left_lane_region_color)
                if is_left_image_white == 2:
                    white_temp_result_test = self.lt_white_lane_processing(left_lane_region_color)
                    self.lt_color = 0
                else:
                    self.lt_color = 0
                    self.lt_inner_percentage = 0
                    self.lt_outer_percentage = 0
            if self.lt_inner_percentage >= 100:
                self.lt_inner_percentage = 99
            if self.lt_outer_percentage >= 100:
                self.lt_outer_percentage = 99
        else:
            self.lt_color = 0
            self.lt_inner_percentage = 0
            self.lt_outer_percentage = 0
        
        is_right_lane_cropped, right_lane_region_gray, right_lane_region_color = \
            self.crop_right_region(bview_gray, bview_color)
        
        if is_right_lane_cropped == 0:
            # cv2.imshow("right_lane_region_color", right_lane_region_color)
            # cv2.waitKey(1)

            is_right_image_yellow = self.yellow_color_verification(right_lane_region_color)
            
            if is_right_image_yellow:
                yellow_temp_result = self.rt_yellow_lane_processing(right_lane_region_color)
                self.rt_color = 1
            else:
                is_right_image_white = self.line_color_verification(right_lane_region_color)
                if is_right_image_white == 2:
                    white_temp_result_test = self.rt_white_lane_processing(right_lane_region_color)
                    self.rt_color = 0
                else:
                    self.rt_color = 0
                    self.rt_inner_percentage = 0
                    self.rt_outer_percentage = 0
            if self.rt_inner_percentage >= 100:
                self.rt_inner_percentage = 99
            if self.rt_outer_percentage >= 100:
                self.rt_outer_percentage = 99
        else:
            self.rt_color = 0
            self.rt_inner_percentage = 0
            self.rt_outer_percentage = 0
        # cv2.waitKey(0)
        
    def lane_line_assessment_pipeline(self, input_image):
        # cv2.imshow("grabbed image", input_image)
        # cv2.waitKey(1)
        
        lane_assessment_input_color = input_image.copy()
        # top_half_removed_color = remove_top_half_image(lane_assessment_input_color)
        lane_assessment_input_gray = self.convert_to_grayscale(lane_assessment_input_color)
        # lane_assessment_input_color = top_half_removed_color.copy()
        
        if self.process_flag == 0:
            self.dashboard_detection(lane_assessment_input_gray)
            if self.dashboard_detected_line != 0:
                print("dashboard_detected_line = ", self.dashboard_detected_line)
                self.crop_top_line = self.dashboard_detected_line - 140 - 2
                self.process_flag = 1
                print("Now process flag is set to 1.")
        elif self.process_flag == 1:
            
            if self.transformation_matrix_init:
                # print("@ transformation_matrix_init phase")
                self.lsd_matrix_verify_function_call_count += 1
                
                if self.lsd_matrix_verify_function_call_count >= 25:
                    self.lsd_matrix_verify_function_call_count = 0
                    self.process_flag = 0
                    self.dashboard_detected_line = 0
                    self.crop_top_line = 0
                else:
                    self.lsd_transformation_matrix_update(lane_assessment_input_gray)
            else:
                
                if self.birdeye_verify_init:
                    self.birdeye_verify_function_call_count += 1
                    if self.birdeye_verify_function_call_count >= 70:
                        # pass
                        self.birdeye_verify_function_call_count = 0
                        self.transformation_matrix_init = True
                        self.transformation_source_point = np.zeros((4, 2), dtype="float32")
                        self.transformation_source_point_scaled = np.zeros((4, 2), dtype="float32")
                        self.right_angle_data.clear()
                        self.left_angle_data.clear()
                        self.compare_x_left = sys.maxsize
                        self.compare_x_right = sys.maxsize * -1
                        self.left_points.clear()
                        self.right_points.clear()
                        self.process_flag = 0
                        self.dashboard_detected_line = 0
                        self.crop_top_line = 0
                    else:
                        self.birdeye_view_verification(lane_assessment_input_color, lane_assessment_input_gray)
                
                else:
                    self.process_flag = 2
                    # birdeye_image = transform_image(lane_assessment_input_color)
        
        elif self.process_flag == 2:
            self.pipeline(input_image)
            self.lane_assessment_log()
            # print(self.m_19_f_07, self.m_19_f_08, self.m_19_f_09, self.m_19_f_10)
            # print("@ process flag : ", self.process_flag)
            # print()
