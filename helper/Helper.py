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

    # 4
    def predict_values_from_model(self, model_id, content):
        try:
            # Encode the testing values
            features_list = [
                "Stroke",
                "Smoker",
                "BMI",
                "PhysActivity",
                "Fruits",
                "PhysHlth",
                "HighBP",
                "DiffWalk",
                "NoDocbcCost",
                "HighChol",
                "GenHlth",
                "MentHlth",
                "Sex",
                "Diabetes",
                "Income",
                "AnyHealthcare",
                "HvyAlcoholConsump",
                "Education",
                "CholCheck",
                "Veggies",
                "Age"
            ]
            lables_list =['HeartDiseaseorAttack']

            testing_values = []
            for i in features_list:
                feature_value = str(content[i])
                final_feature_value = feature_value  # float(feature_value) if feature_value.isnumeric() else feature_value
                testing_values.append(final_feature_value)

            # ------------------Predict values from the model-------------------------#
            model = pickle.load(open('pkls/' + str(model_id) + '/' + str(model_id) + '_model.pkl', 'rb'))

            testing_values = numpy.array(testing_values)
            encode_df_testing_values = [testing_values]

            # Sclaing testing values
            scalar_file_name = 'pkls/' + str(model_id) + '/' + str(model_id) + '_scalear.sav'
            s_c = pickle.load(open(scalar_file_name, 'rb'))
            test_x = s_c.transform(encode_df_testing_values)

            predicted_values = [model.predict(test_x)]
            decoded_predicted_values = predicted_values

            predicted_values_json = {}
            for j in range(len(decoded_predicted_values)):
                for i in range(len(lables_list)):
                    bb = decoded_predicted_values[j][i]
                    predicted_values_json[lables_list[i]] = round(decoded_predicted_values[j][i])
                    # NpEncoder = NpEncoder(json.JSONEncoder)
                json_data = json.dumps(predicted_values_json, cls=NpEncoder)

            return json_data

        except Exception as e:
            return [
                'Not able to predict. One or more entered values has not relevant value in your dataset, please enter data from provided dataset']
