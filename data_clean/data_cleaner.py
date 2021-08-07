# -*- coding: utf-8 -*-
import re
import numpy as np
import pandas as pd
import time
import pickle
import datetime

def guess_date(string):
    for fmt in ['%d-%b-%y',"%Y/%m/%d", "%d-%m-%Y", "%Y%m%d"]:
        try:
            return pd.to_datetime(string, format = fmt)
        except ValueError:
            continue
    raise ValueError(string)


class DataCleaner:
    def __init__(self, factiva_df, frequency : str):
        # frequency can choose 'M' or 'D'
        self.factiva_series = factiva_df.loc[:, ['pd', 'lp', 'td', 'an']] # date, news article,accession number
        self.frequency = frequency
        self.m_data_series = pd.Series()
        self.find_body = re.compile('"body": "(.*?)", "mimeType"')
        self.find_time = re.compile('"versionCreated": "(.*?)"')
        self.find_source_text_for_eikon = re.compile(r'source text for eikon:.*$', re.IGNORECASE)
        self.find_source_link = re.compile(r'-- source link:.*$', re.IGNORECASE)
        self.find_source_text_in = re.compile(r'source text in.*$', re.IGNORECASE)
        self.find_source_text = re.compile(r'source text -.*$', re.IGNORECASE)
        self.find_keywords = re.compile(r'keywords:.*$', re.IGNORECASE)
        self.find_bracket = re.compile(r'\[.*?\]')
        self.find_angle_quotation = re.compile(r'<.*?>')
        self.find_header = re.compile(r'^.* - ')
        self.find_fitch = re.compile(r"(See Fitch's recent commentary.*$)", re.IGNORECASE)
        self.find_fitch2 = re.compile(r"(contact: .*$)", re.IGNORECASE)
        self.find_fitch3 = re.compile(r"(https://www.fitchratings.com/site.*$)", re.IGNORECASE)
        self.find_fitch4 = re.compile(r"(media relations: .*$)", re.IGNORECASE)
        self.find_note = re.compile(r"(note: .*$)", re.IGNORECASE)
        self.find_price_table = re.compile(r"(Reuters Terminal users can see .*$)", re.IGNORECASE)
        self.find_price_table2 = re.compile(r"(palm, soy and crude oil prices at \d+ gmt.*$)", re.IGNORECASE)
        self.find_change = re.compile(r"(\..*?change on the day.*$)", re.IGNORECASE)
        self.invalid_headline = re.compile(r'headline": "TABLE-" '
                                           r'| "headline": "*TOP NEWS*" '
                                           r'| "headline": "DIARY-" '
                                           r'| "headline": "SHH .* Margin Trading" '
                                           r'| "headline": "North American power transmission outage update - PJM" '
                                           r'| "headline": "UPDATE 1"')
        self.unique_alt_id = set()


    def __call__(self):
        self.factiva_series['news'] = self.factiva_series['lp'] + self.factiva_series['td']
        self.factiva_series = self.factiva_series.loc[:, ['pd', 'news', 'an']].dropna(axis=0,how='any')
        self.factiva_series.columns = ['date', 'news', 'number']
        self.factiva_series['news'] = self.factiva_series['news'].apply(lambda x: self.clean_up(x))
        self.factiva_series['date'] = guess_date(self.factiva_series['date'])
        self.factiva_series = self.factiva_series.sort_values('date')

        if self.frequency == 'M':
            self.factiva_series = pd.Series(self.factiva_series['news'].values,
                                            index=self.factiva_series['date'].apply(lambda x:np.datetime64(x,'M')))
            # 看怎么修正成可以适应不同str格式转化
            '''
            self.factiva_series = pd.Series(self.factiva_series['news'].values,
                                            index=pd.to_datetime(self.factiva_series['date'], format='%d %B %Y'))
            '''
            #self.factiva_series = pd.Series(self.factiva_series['news'].values, index = pd.to_datetime(self.factiva_series['date'], format='%d-%b-%y'))
            for date in self.factiva_series.index:
                temp_text = self.factiva_series[date]
                date = np.datetime64(date,'M')
                if isinstance(temp_text,str):  # string, only one value
                    self.m_data_series.at[date] = [temp_text]
                else:    # Series, more than 1 text,transfer to a list contain more than 1 element
                    self.m_data_series.at[date] = list(temp_text.values.flatten())


        else:
            self.factiva_series = pd.Series(self.factiva_series['news'].values,
                                            index=pd.to_datetime(self.factiva_series['date'], format='%d %B %Y'))
            for date in self.factiva_series.index:
                temp_text = self.factiva_series[date]
                if isinstance(temp_text, str):
                    self.m_data_series.at[date] = [temp_text]
                else:
                    self.m_data_series.at[date] = list(temp_text.values.flatten())


        #self.factiva_series['date'] = pd.to_datetime(self.factiva_series['date'] ,format='%d %B %Y')
        #self.factiva_df['date'] = pd.to_datetime(self.factiva_df['date'] ,unit = 'D')



    def clean_up(self, data_line):
        cleaned = self.data_clean(data_line)
        return cleaned


    def data_clean(self, target: str):
        """
        Function which will do the data clean
        Remove the parentheses
        :param target: the input string data
        :return: string after data clean
        """

        target = self.remove_header(target)
        target = self.remove_price_table(target)
        target = self.remove_keywords(target)
        target = self.remove_source_link(target)
        target = self.remove_source_text_for_eikon(target)
        target = self.remove_source_text_in(target)
        target = self.remove_source_text(target)
        target = self.remove_brackets(target)
        target = self.remove_fitch(target)
        target = self.remove_fitch2(target)
        target = self.remove_fitch3(target)
        target = self.remove_fitch4(target)
        target = self.remove_note(target)
        target = self.remove_price_table2(target)
        target = self.remove_change(target)
        target = target.replace('\\n', ' ') \
            .replace('\\\"', '') \
            .replace('\\r', ' ') \
            .replace('*', '') \
            .replace('“', '') \
            .replace('”', '')
        return target.lower().strip()

    @staticmethod
    def remove_nested_parentheses(data: str):
        """
        :param data: input string
        :return: data with all parentheses removed
        """
        result = ''
        depth = 0
        for letter in data:
            if letter == '(':
                depth += 1
            elif letter == ')':
                depth -= 1
            elif depth == 0:
                result += letter
        return result

    @staticmethod
    def remove_nested_brackets(data: str):
        """
        :param data: input string
        :return: data with all parentheses removed
        """
        result = ''
        depth = 0
        for letter in data:
            if letter == '[':
                depth += 1
            elif letter == ']':
                depth -= 1
            elif depth == 0:
                result += letter
        return result

    @staticmethod
    def remove_nested_square_brackets(data: str):
        """
        :param data: input string
        :return: data with all parentheses removed
        """
        result = ''
        depth = 0
        for letter in data:
            if letter == '{':
                depth += 1
            elif letter == '}':
                depth -= 1
            elif depth == 0:
                result += letter
        return result

    def remove_keywords(self, data):
        return self.find_keywords.sub(r'', data)

    def remove_source_link(self, data):
        return self.find_source_link.sub(r'', data)

    def remove_fitch(self, data):
        return self.find_fitch.sub(r'', data)

    def remove_fitch2(self, data):
        return self.find_fitch2.sub(r'', data)

    def remove_fitch3(self, data):
        return self.find_fitch3.sub(r'', data)

    def remove_fitch4(self, data):
        return self.find_fitch4.sub(r'', data)

    def remove_note(self, data):
        return self.find_note.sub(r'', data)

    def remove_price_table(self, data):
        return self.find_price_table.sub(r'', data)

    def remove_price_table2(self, data):
        return self.find_price_table2.sub(r'', data)

    def remove_change(self, data):
        return self.find_change.sub(r'', data)

    def remove_source_text_for_eikon(self, data):
        return self.find_source_text_for_eikon.sub(r'', data)

    def remove_source_text_in(self, data):
        return self.find_source_text_in.sub(r'', data)

    def remove_source_text(self, data):
        return self.find_source_text.sub(r'', data)

    def remove_brackets(self, data):
        """
        Remove ((.*)) and [.*] and <.*> and nested parentheses
        :param data:  Input string data
        :return: String data after processing.
        """
        data = self.find_bracket.sub(r'', data)
        data = self.find_angle_quotation.sub(r'', data)
        data = self.remove_nested_parentheses(data)
        data = self.remove_nested_brackets(data)
        data = self.remove_nested_square_brackets(data)
        return data

    def remove_header(self, data):
        data = self.find_header.sub(r'', data)
        return data

    def date_transfer(self,date_str):
        data = pd.to_datetime(date_str,format='%d %B %Y')
        return data