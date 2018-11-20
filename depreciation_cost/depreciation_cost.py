# coding: utf-8
import pandas as pd
import datetime
import sys

column_dict = {
    '자산번호':'id',
    '보조번호':'sub_id',
    '자산클래스':'asset_class_id',
    '자산클래스명':'asset_class_name',
    '자산내역':'asset_name',
    '분류':'cate_id',
    '분류명':'cate_name',
    '코스트센터':'depart_id',
    '코스트센터명':'depart_name',
    '수  량': 'val',
    '자본화일': 'intro_date',
    '총취득원가': 'original_price',
}

column_dict_rev = {
    'id': '자산번호',
    'sub_id': '보조번호',
    'asset_class_id': '자산클래스',
    'asset_class_name': '자산클래스명',
    'asset_name': '자산내역',
    'cate_id': '분류',
    'cate_name': '분류명',
    'depart_id': '코스트센터',
    'depart_name': '코스트센터명',
    'val':'수  량',
    'intro_date': '자본화일',
    'original_price': '총취득원가',
}

now = datetime.datetime.now()
current_year = now.year
current_month = now.month


def reset_intro_date(row):
    intro_date = row.intro_date
    row['intro_year'] = int(intro_date[:4])
    row['intro_month'] = int(intro_date[5:7]) 
    return row


def get_depre_month(row, std_year):
    depre_month = 0
    if current_year < std_year:
        if row['expired_year'] < std_year:
            depre_month = 0
        elif row['expired_year'] > std_year:
            depre_month = 12
        elif row['expired_year'] == std_year:
            depre_month = row['intro_month'] - 1 
    elif current_year == std_year:
        if row['expired_year'] < std_year:
            depre_month = 0
        elif row['expired_year'] > std_year:
            depre_month = 12 - current_month + 1
        elif row['expired_year'] == std_year:
            if row['intro_month'] < current_month:
                depre_month = 0
            else:
                depre_month = row['intro_month'] - current_month         
    return depre_month


if __name__ == '__main__':
    std_year = sys.argv[1]
    df = pd.read_excel('./depreciation_cost.xlsx')

    df.columns = [col.strip() for col in df.columns]
    df = df.rename(column_dict, axis=1)
    df = df.loc[:, column_dict.values()]
    df_depre = df.loc[:, ['intro_date', 'original_price']]
    df_depre = df_depre.apply(lambda row: reset_intro_date(row), axis=1)

    df_depre['expired_year'] = df_depre['intro_year']+4
    df_depre['monthly_cost'] = df_depre['original_price']/48
    df_depre['depre_month'] = df_depre.apply(lambda row: get_depre_month(row, std_year=std_year), axis=1)
    df_depre['depreciation_cost'] = df_depre.apply(lambda row: row['monthly_cost'] * row['depre_month'], axis=1)
    df['depreciation_cost'] = df_depre['depreciation_cost'].tolist()

    print(df['depreciation_cost'].sum())

    df = df.rename(column_dict_rev, axis=1)
    df.to_excel('./depreciation_cost_result.xlsx', index=False)
