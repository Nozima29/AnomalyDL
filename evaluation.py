class Single_Period:
    def __init__(self, first, last, name):
        self._first_timestamp = first
        self._last_timestamp = last
        self._name = name

    def set_time(self, first, last):
        self._first_timestamp = first
        self._last_timestamp = last

    def get_time(self):
        return self._first_timestamp, self._last_timestamp

    def set_name(self, str):
        self._name = str

    def get_name(self):
        return self._name

    def __eq__(self, other):
        return self._first_timestamp == other.get_time()[0] and self._last_timestamp == other.get_time()[1]


class Evaluation:
    def __init__(self):
        self._prediction = []  # list
        self._ground_truth = []  # list

        self._set_prediction = False
        self._set_ground_truth = False
        pass

    def set_prediction(self, prediction):
        self._prediction = self._head_tail_index(prediction)
        self._set_prediction = True

    def _head_tail_index(self, prediction):
        return_list = []
        start_idx = -1

        if prediction[0] == 1:
            start_idx = 0

        for idx in range(1, len(prediction)):
            if prediction[idx] == 1 and prediction[idx - 1] == 0:
                start_idx = idx

            if prediction[idx] == 0 and prediction[idx - 1] == 1:
                return_list.append([start_idx, idx - 1])

        if prediction[len(prediction) - 1] == 1:
            return_list.append([start_idx, len(prediction) - 1])

        return return_list

    def get_n_predictions(self):
        return len(self._prediction)

    def set_ground_truth(self, ground_truth_list):
        self._ground_truth = []
        for single_answer in ground_truth_list:
            self._ground_truth.append(Single_Period(
                single_answer[0], single_answer[1], str(single_answer[2])))

        self._set_ground_truth = True

    def _extend_single_truth(self, single_list, extend_head, extend_tail):
        for idx in range(len(single_list)):
            if idx != 0:
                first = max(single_list[idx].get_time()[
                            0] - extend_head, single_list[idx-1].get_time()[1])
            else:
                first = single_list[idx].get_time()[0] - extend_head

            if first < 0:
                first = 0

            if idx != len(single_list)-1:
                last = min(single_list[idx].get_time()[
                           1] + extend_tail, single_list[idx+1].get_time()[0])
            else:
                last = single_list[idx].get_time()[1] + extend_tail
            single_list[idx].set_time(first, last)

    def extend_ground_truth(self, extend_head, extend_tail):
        self._extend_single_truth(self._ground_truth, extend_head, extend_tail)

    def _is_intersection(self, prediction_term, attack_term):
        if prediction_term[1] >= attack_term[0] and attack_term[1] >= prediction_term[0]:
            return True
        else:
            return False

    def _find_hit_lists(self, intersect_ratio, multi_hit=True):
        hit_ground_truth = []
        hit_prediction = []

        for idx in range(len(self._ground_truth)):
            hit_ground_truth.append(False)
        for idx in range(len(self._prediction)):
            hit_prediction.append(False)

        for idx_gt in range(len(self._ground_truth)):
            for idx_pr in range(len(self._prediction)):
                length = self._intersect_len(
                    self._ground_truth[idx_gt], self._prediction[idx_pr])
                term_len = self._prediction[idx_pr][1] - \
                    self._prediction[idx_pr][0] + 1
                if float(length/term_len) >= intersect_ratio:
                    if multi_hit == True or hit_prediction[idx_pr] == False:
                        hit_ground_truth[idx_gt] = True
                    hit_prediction[idx_pr] = True

        return hit_prediction, hit_ground_truth

    def detected_attacks(self, intersect_ratio, multi_hit=True):
        assert (self._set_ground_truth ==
                True and self._set_prediction == True)

        hit_ground_truth = []
        hit_prediction = []

        for idx in range(len(self._ground_truth)):
            hit_ground_truth.append(False)
        for idx in range(len(self._prediction)):
            hit_prediction.append(False)

        for idx_gt in range(len(self._ground_truth)):
            for idx_pr in range(len(self._prediction)):
                length = self._intersect_len(
                    self._ground_truth[idx_gt].get_time(), self._prediction[idx_pr])
                term_len = self._prediction[idx_pr][1] - \
                    self._prediction[idx_pr][0] + 1
                if float(length/term_len) >= intersect_ratio:
                    if multi_hit == True or hit_prediction[idx_pr] == False:
                        hit_ground_truth[idx_gt] = True
                    hit_prediction[idx_pr] = True

        found_cnt = 0
        found_str = ' - Found: '
        notfound_str = ' - Not Found: '
        for idx in range(len(hit_ground_truth)):
            if hit_ground_truth[idx] == True:
                found_str += self._ground_truth[idx].get_name() + ', '
                found_cnt += 1
            else:
                notfound_str += self._ground_truth[idx].get_name() + ', '

        print('#Found attacks: ' + str(found_cnt) +
              '/' + str(len(self._ground_truth)))
        print('Detecting Ratio: ' +
              str(round(float(found_cnt)/len(self._ground_truth), 2)))
        print(found_str[:-2])
        print(notfound_str[:-2])


if __name__ == '__main__':
    eval = Evaluation()
    eval.set_prediction(predict_lists[0])
    eval.set_ground_truth(attack_lists[0])
    eval.extend_ground_truth(extend_head=extend_head, extend_tail=extend_tail)

    print(head_str[0])
    #print('#Predictions: ' + str(eval.get_n_predictions()))
    eval.detected_attacks(correct_ratio_d, multi_hit)
    eval.false_alarm(correct_ratio_f, multi_hit)
    print('', flush=True)