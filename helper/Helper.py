import json
import pickle

import numpy as numpy

from helper.NpEncoder import NpEncoder


class Helper:

    # 1
    def classify_data(self, content, model_id):
        testing_values = []
        features_list = ['data']
        for i in features_list:
            feature_value = str(content[i])
            final_feature_value = feature_value  # float(feature_value) if feature_value.isnumeric() else feature_value
            testing_values.append(final_feature_value)
        text_category = [numpy.array(self.classify_text(feature_value, model_id))]

        # Create predicted values json object
        lables_list = ['category']
        text_category_json = {}
        for j in range(len(text_category)):
            for i in range(len(lables_list)):
                bb = text_category[j][i]
                text_category_json[lables_list[i]] = text_category[j][i]
                # NpEncoder = NpEncoder(json.JSONEncoder)
            json_data = json.dumps(text_category_json, cls=NpEncoder)

        return json_data

    # 2
    def classify_text(self, test_text, model_name=''):
        return self.classify(test_text, model_name)

    # 3
    def classify(self, text, file_name=''):
        # Load model
        clf_filename = '%s%s%s' % ('pkls/', file_name, '/classifier_pkl.pkl')
        np_clf = pickle.load(open(clf_filename, 'rb'))

        # load vectorizer
        vec_filename = '%s%s%s' % ('pkls/', file_name, '/vectorized_pkl.pkl')
        vectorizer = pickle.load(open(vec_filename, 'rb'))

        pred = np_clf.predict(vectorizer.transform([text]))

        return pred