from typing import Union
from datasets.ds_coco import CocoDataProvider


class DatasetFactory(object):
    """
    Dataset simple factory method
    """

    @staticmethod
    def create(params) -> CocoDataProvider:
        """
        Creates Dataset based on detector type
        :param params: Dataset settings
        :return: Dataset instance. In case of unknown Dataset type throws exception.
        """

        if params['DATASET']['name'] == 'coco':
            return CocoDataProvider(params['DATASET']['path'],
                              batch_sizes=params['DATASET']['batch_sizes'],
                              tiny=params['DATASET']['tiny'],
                              transform_keys=params['DATASET']['transforms'],
                              num_workers=params['DATASET']['num_workers']
                              )

        raise ValueError("DatasetFactory(): Unknown Dataset type: " + params['Dataset']['type'])
