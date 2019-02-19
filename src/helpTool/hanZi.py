# encoding: UTF-8


class HanZi:
    """
    字典内部对象
    同一个汉字存储在同一个文件夹下面
    """
    def __init__(self,
                 folder_name,
                 index):
        self._folder_name = folder_name
        self._index = index

    @property
    def folder_name(self):
        return self._folder_name

    @property
    def index(self):
        return self._index

    def get_and_increase_index(self):
        rtn = self._index
        self._index += 1
        return rtn
