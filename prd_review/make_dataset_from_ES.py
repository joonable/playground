import pandas as pd

from elasticsearch5 import Elasticsearch

#pid = 2337

es = Elasticsearch(hosts=ES_HOST)
count = es.count(index="prd_review")['count']


def get_mtermvectors(ids):
    body = dict()
    body["ids"] = ids
    body["parameters"] = {"fields": ["title"]}

    res = es.mtermvectors(index='prd_review', doc_type='_doc', body=body)['docs']
    return res


def get_termvectors(id):
    res = es.termvectors(index='prd_review', doc_type='_doc', id=id)['term_vectors']
    if 'title' in res.keys():
        return res
    else:
        return None


def sort_terms_vector(term_vectors):
    if not term_vectors:
        return None
    term_dict = {}
    for term, val in term_vectors[0].items():
        for pos_info in val['tokens']:
            term_dict[pos_info['position']] = term
    sorted_terms = sorted(term_dict.items())
    sorted_terms = [tup[1] for tup in sorted_terms]
    return sorted_terms


if __name__ == '__main__':
    count_list = [x for x in range(0, count, 10000)]
    count_list.append(count)

    results = list()
    results.append(es.search(index='prd_review', size=10000, scroll='1m'))
    scroll_id = results[0]['_scroll_id']
    results = results[0]['hits']['hits']

    for _ in range(count // 10000):
        results.extend(es.scroll(scroll_id=scroll_id, scroll='1m')['hits']['hits'])

    results = [result['_source'] for result in results]

    data = []
    for result in results:
        data.append({})
        data[-1]['m_id'] = result['message_id']
        data[-1]['score'] = result['prd_satisfact']
        data[-1]['cus_grade'] = result['cus_grade']
        data[-1]['best_flag'] = result['best_flag']
        data[-1]['text'] = result['title']

    df = pd.DataFrame(data)
    df.m_id = df.m_id.astype('str')
    # df['pos'] = df.id.apply(lambda _id: get_termvectors(_id))
    df = df.set_index('m_id')

    df['term_vectors'] = None
    for idx in range(len(count_list) - 1):
        ids = df.iloc[count_list[idx]:count_list[idx + 1]].index.tolist()
        term_list = get_mtermvectors(ids)
        ids = []
        temp = []
        for x in term_list:
            ids.append(x['_id'])
            if 'title' in x['term_vectors'].keys():
                temp.append([x['term_vectors']['title']['terms']])
            else:
                temp.append(None)
        df.loc[ids, 'term_vectors'] = temp

    df['sorted_terms'] = df.term_vectors.apply(lambda x: sort_terms_vector(x))
    df = df.drop(['term_vectors'], axis=1).reset_index()
    df.to_pickle('prd_sample.pkl')
