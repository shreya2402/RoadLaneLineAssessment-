import time
import constant
import cv2
import lane_line_assessment


def read_list_logs():
    """

    :return:
    """
    # Path to log_list.rlog
    list_dir = constant.list_dir
    path_to_data = constant.path_to_data
    file = open(list_dir, 'r')
    data = file.read().split('\n')
    file.close()
    logs_path = []
    for l in data:
        mac = l.split('_')[0]  # Mac Address
        if l.endswith('.rlog'):
            logs_path.append(path_to_data + mac + '/' + l)
        else:
            continue
    return logs_path


def sort_list(list_input):
    # Sorting list in Ascending Order
    list_input.sort()


def get_mac_log_id(log):
    """
    :param log:
    :return: mac, log_id
    """
    log_name = log.split('/')[2]
    mac = log_name.split('_')[0]
    log_id = log_name.split('_')[1].split('.')[0]
    return mac, log_id


def get_18_entries_from_log(log):
    """
        function returns log with 71 entry. Whenever there is change in lat/long of 72 entry a 71 entry is added.
        Other 72 entries are discarded.
        :param log:
        :return: log_with_71
        """
    log_file = open(log, 'r')
    log_data = log_file.read().split('\n')
    log_file.close()
    log_data = [i for i in log_data if i[:2] == "18"]
    return log_data


def read_image(image_path, image_name):
    grabbed_image = cv2.imread(image_path + image_name)
    is_image_valid = True
    if grabbed_image is None:
        is_image_valid = False
    return is_image_valid, grabbed_image


def main():
    # read log path using read_list_logs.
    # Here no need to pass any input arguments as folder structure is constant So file can be opened by relative paths.
    log_paths = read_list_logs()
    sort_list(log_paths)
    processing_log_number = 0

    for log in log_paths:
        obj_lane_line_assessment = lane_line_assessment.LaneLineAssessment()
        processing_log_number += 1
        print("Processing log number : ", processing_log_number, "\nName :", log, "\n")
        mac, log_id = get_mac_log_id(log)
        #  call find_infra_location function to find locations of infra in logs files.
        log_18_entries = get_18_entries_from_log(log)
        image_path = constant.path_to_images + mac + "/"
        out_log = []
        # break_count = 0
        image_count = 0
        print("Number of images are : ", len(log_18_entries))
        for log_18_sub_entry in log_18_entries:

            # break_count += 1
            # if break_count > 200:
            #     break
            log_18_sub_entry_split = log_18_sub_entry.split(",")
            image_name = mac + "_" + log_18_sub_entry_split[2] + ".jpg"
            # print(image_count, " | ", image_name)
            image_count += 1
            is_image_valid, grabbed_image = read_image(image_path, image_name)
            
            if image_count % 100 == 0:
                print(image_count, " | ", image_name)
            if is_image_valid:
                obj_lane_line_assessment.lane_line_assessment_pipeline(grabbed_image)
            else:
                print("Image not found : ", image_path + image_name)
                
            out_string = \
                obj_lane_line_assessment.m_19_f_01 + "," + \
                log_18_sub_entry_split[5] + "," + \
                log_18_sub_entry_split[6] + "," + \
                "-1" + "," + \
                log_18_sub_entry_split[2] + "," + \
                "-1" + "," + \
                obj_lane_line_assessment.m_19_f_07 + "," + \
                obj_lane_line_assessment.m_19_f_08 + "," + \
                obj_lane_line_assessment.m_19_f_09 + "," + \
                obj_lane_line_assessment.m_19_f_10 + "," + \
                log_18_sub_entry_split[3] + "," + \
                "data/" + mac + "/" + mac + "_" + log_18_sub_entry_split[2] + ".jpg" + "," +\
                mac + "," + \
                "-1" + "," + \
                "-1" + "," + \
                obj_lane_line_assessment.m_19_overall_score + "\n"
            out_log.append(out_string)
        
        print("@ writing log file output. Named as : ", log.split(".rlog")[0] + "_lane_line" + ".rlog")
        with open(log.split(".rlog")[0] + "_lane_line" + ".rlog", "w") as save_file:
            for entry in out_log:
                save_file.write(entry)
                
        # print()


if __name__ == '__main__':
    start_time = time.time()
    main()
    print('Process completed in %.2f seconds' % (time.time() - start_time))
