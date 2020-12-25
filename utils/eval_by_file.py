import json


def eval(predicted_file, gt_file):

    mistakes_samples = {}

    with open(predicted_file, "r") as f:
        prediction = json.load(f)

    with open(gt_file, "r") as f:
        gt = json.load(f)

    for question_id in prediction.keys():
        pre_answer = prediction[question_id][0]
        gt_ans = gt[question_id]
        if pre_answer != gt_ans:
            mistakes_samples[question_id] = gt[question_id]
            mistakes_samples[question_id].update({"prediction":prediction[question_id]})

    json_obj = json.dump(mistakes_samples)

    with open("pre_analysis.json","w") as outfile:
        outfile.write(json_obj)

