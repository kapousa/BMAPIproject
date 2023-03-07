class SystemHelper:

    def encode_input_values(self, model_id, features_list, input_values):
        encoded_columns = []
        model_encoded_columns = numpy.array(
            ModelEncodedColumns.query.with_entities(ModelEncodedColumns.column_name).filter(
                and_(ModelEncodedColumns.model_id == str(model_id),
                     ModelEncodedColumns.column_type == 'F')).all())
        model_encoded_columns = model_encoded_columns.flatten()
        for i in range(len(input_values)):
            input_value = input_values[i].strip()
            if (not input_value.isdigit()) and (features_list[i] in model_encoded_columns):
                col_name = features_list[i]
                pkl_file_location = self.pkls_location + str(model_id) + '/' + col_name + '_pkle.pkl'
                encoder_pkl = pickle.load(open(pkl_file_location, 'rb'))
                column_data_arr = numpy.array(input_value)
                encoded_values = encoder_pkl.transform(column_data_arr.reshape(-1, 1)) if (
                            Helper.is_time(col_name) == 0) else self.endcode_datetime_column(col_name,
                                                                                             column_data_arr.flatten())
                encoded_columns.append(encoded_values[0])
            else:
                encoded_columns.append(input_value)

        return [encoded_columns]

    def decode_output_values(self, model_id, labels_list, input_values):
        input_values = numpy.array(input_values)
        input_values = input_values.flatten()
        input_values = input_values.reshape(1, len(input_values))
        decoded_results = []
        decoded_row = []
        model_encoded_columns = numpy.array(
            ModelEncodedColumns.query.with_entities(ModelEncodedColumns.column_name).filter(
                and_(ModelEncodedColumns.model_id == str(model_id),
                     ModelEncodedColumns.column_type == 'L')).all())
        model_encoded_columns = model_encoded_columns.flatten()
        for i in range(len(input_values)):
            input_values_row = input_values[i]
            for j in range(len(input_values[i])):
                if labels_list[j] in model_encoded_columns:
                    col_name = labels_list[j]
                    pkl_file_location = self.pkls_location + str(model_id) + '/' + col_name + '_pkle.pkl'
                    encoder_pkl = pickle.load(open(pkl_file_location, 'rb'))
                    column_data_arr = numpy.array(input_values_row[j], dtype='int')
                    original_value = encoder_pkl.inverse_transform(column_data_arr.reshape(-1, 1)) if (
                                Helper.is_time(col_name) == 0) else self.decode_datetime_value(
                        column_data_arr)
                    decoded_row.append(original_value[0].strip())
                else:
                    decoded_row.append(str(input_values_row[j]))
            decoded_results.append(decoded_row)
        return np.array(decoded_results)