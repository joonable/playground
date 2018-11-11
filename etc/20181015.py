import pandas as pd

df = pd.DataFrame(
    data = [{"name": "남일우", "sex": "m", "department": "IT개발팀"},
     {"name": "이수미", "sex": "f", "department": "IT개발팀"},
     {"name": "정재현", "sex": "m", "department": "영상제작1팀"},
     {"name": "박수정", "sex": "f", "department": "고객서비스팀"},
     {"name": "이규봉", "sex": "m", "department": "트랜드패션팀"},
     {"name": "박문식", "sex": "m", "department": "DM팀"},
     {"name": "선수현", "sex": "f", "department": "디지털 엑셀러레이션팀"},
     {"name": "이승준", "sex": "m", "department": "COE"},
     {"name": "정은성", "sex": "m", "department": "microSVC"}]
    , index=[0,1,2,3,4,5,6,7,8]
)

df_2 = pd.DataFrame(
    data={"name": ["남일우", "이수미", "정재현", "박수정",
                   "이규봉", "박문식", "선수현", "이승준", "정은성"],
          "sex": ["m", "f", "m", "f", "m", "m", "f", "m", "m"],
          "department": ["IT개발팀", "IT개발팀", "영상제작1팀",
                         "고객서비스팀", "트랜드패션팀", "DM팀",
                         "디지털 엑셀러레이션팀", "COE", "microSVC"]}
    # , index=[0,1,2,3,4,5,6,7,8]
)

df_3 = pd.DataFrame(
    data=[["남일우", "m", "IT개발팀"],
          ["이수미", "f", "IT개발팀"],
          ["정재현", "m", "영상제작1팀"],
          ["박수정", "f", "고객서비스팀"],
          ["이규봉", "m", "트랜드패션팀"],
          ["박문식", "m", "DM팀"],
          ["선수현", "f", "디지털 엑셀러레이션팀"],
          ["이승준", "m", "COE"],
          ["정은성", "m", "microSVC"]],
    columns=["name", "sex", "index"],
    index=[0,1,2,3,4,5,6,7,8]
)

square = 0
number = 1

while square < 99:
    square = number ** 2
    print(square)
    number += 1

while True:
    if square < 99:
        break
    square = number ** 2
    print(square)
    number += 1


def sum_two_numbers(a, b):
    val = a + b
    return val

# c = sum_two_numbers(3, 12)
# 실제로 벌어지는 일
(a, b) = (3, 12)
val = a + b
c = val

any_list = [12, -32, 231, 0, -325, 125]


type(df.index)
type(df.columns)
# name, sex, department,
# 남일우, m, IT개발팀,
# 이수미, f, IT개발팀,
# 정재현, m, 영상제작1팀,
# 박수정, f, 고객서비스팀,
# 이규봉, m, 트랜드패션팀,
# 박문식, m, DM팀,
# 선수현, f, 디지털 엑셀러레이션팀,
# 이승준, m, COE,
# 정은성, m, microSVC
#
#
# {"name": "남일우", "sex": "m", "department": "IT개발팀"}
# {"name": "이수미", "sex": "f", "department": "IT개발팀"}
# {"name": "정재현", "sex": "m", "department": "영상제작1팀"}
# {"name": "박수정", "sex": "f", "department": "고객서비스팀"}
# {"name": "이규봉", "sex": "m", "department": "트랜드패션팀"}
# {"name": "박문식", "sex": "m", "department": "DM팀"}
# {"name": "선수현", "sex": "f", "department": "디지털 엑셀러레이션팀"}
# {"name": "이승준", "sex": "m", "department": "COE"}
# {"name": "정은성", "sex": "m", "department": "microSVC"}

# df
# df['name']
# df['department']
# df['sex']
#
# df.index
# df.columns
#
# df.loc[[2, 4, 5, 7], ['name', 'sex']]
# df.iloc[[2, 4, 5, 7], [0, 2]]
#
# df.loc[:, 'name']
# df.iloc[:, 0]
#
# df.loc[8], df.loc[8. :]
# df.iloc[8], df.iloc[8, :]
#
#
# # 특정한 컬럼에서 some_value을 가진 값만 뽑아내기
# df.loc[df['column_name'] == some_value]
# df.loc[df['column_name'] != some_value]
#
# # 특정한 컬럼에서 some_values(복수)를 가진 값만 뽑아내기
# df.loc[df['column_name'].isin(some_values)]
# df.loc[~df['column_name'].isin(some_values)]
#
# # 한번에 여러 개의 조건을 적용시켜 뽑아내기
# df.loc[(df['column_name'] == some_value) & df['other_column'].isin(some_values)]
# df.loc[df['column_name'] == some_value].loc[df['other_column'].isin(some_values)]
#
#
# # isin returns a boolean Series, so to select rows whose value is not in some_values, negate the boolean Series using ~:
#
#
#
# df.loc[df['sex'] == 'm']
# df.loc[df['sex'] != 'f']
# df.iloc[:, 0]
# df.loc[df['sex'] == 'm', :]
#
#
# df.loc[df['department'].isin(['IT개발팀', 'DM팀', 'COE', 'microSVC']), ['department', 'name']]
# df.loc[~df['department'].isin(['영상제작1팀', '고객서비스팀', '트랜드패션팀', '디지털 엑셀러레이션팀']), ['department', 'name']]
#
# df.loc[df['department'].isin(['IT개발팀', '고객서비스팀']) & (df['sex'] == 'f'), ['name']]
# df.loc[df['department'].isin(['IT개발팀', '고객서비스팀'])].loc[df['sex'] == 'f'][['name']]
#
# 'column_name'
# [3]

col_dict = {"name": "이름", "department": "부서", "untitled" : "무제"}
index_dict = {0: "zero", 4: "four", 10: "ten"}
df = df.rename(col_dict, axis=1)
df = df.rename(index_dict, axis=0)

string = ' xoxo love xoxo   '

# Leading whitepsace are removed
print(string.strip())


df['new_col'] = df['col_name'].apply(lambda x: x)

new_list = []
for x in any_list:
    new_list.append(x)
