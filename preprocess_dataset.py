import pickle
import json
import pandas as pd

df_sample = pd.read_excel('./sample_v2.xlsx')

lst_sentences = df_sample['가/을/이'].values.tolist()
df_sample['label_가'] = df_sample['label_가'].astype(str)
lst_label = df_sample['label_가'].values.tolist()

lst_zip = list(map(list,zip(lst_sentences, lst_label)))

print(lst_zip[0])

with open("./ga_dataset_test.txt", "wb") as fp:
    pickle.dump(lst_zip, fp)

lst_sentences = df_sample['만'].values.tolist()
df_sample['label_만'] = df_sample['label_만'].astype(str)
lst_label = df_sample['label_만'].values.tolist()

lst_zip = list(map(list,zip(lst_sentences, lst_label)))

print(lst_zip[0])

with open("./man_dataset_test.txt", "wb") as fp:
    pickle.dump(lst_zip, fp)

lst_sentences = df_sample['은/는'].values.tolist()
df_sample['label_은'] = df_sample['label_은'].astype(str)
lst_label = df_sample['label_은'].values.tolist()

lst_zip = list(map(list,zip(lst_sentences, lst_label)))

print(lst_zip[0])

with open("./eun_neun_dataset_test.txt", "wb") as fp:
    pickle.dump(lst_zip, fp)

lst_sentences = df_sample['도'].values.tolist()
df_sample['label_도'] = df_sample['label_도'].astype(str)
lst_label = df_sample['label_도'].values.tolist()

lst_zip = list(map(list,zip(lst_sentences, lst_label)))

print(lst_zip[0])

with open("./do_dataset_test.txt", "wb") as fp:
    pickle.dump(lst_zip, fp)

lst_sentences = df_sample['부사_매우'].values.tolist()
df_sample['label_매우'] = df_sample['label_매우'].astype(str)
lst_label = df_sample['label_매우'].values.tolist()

lst_zip = list(map(list,zip(lst_sentences, lst_label)))

print(lst_zip[0])

with open("./very_dataset_test.txt", "wb") as fp:
    pickle.dump(lst_zip, fp)

lst_sentences = df_sample['부사_약간'].values.tolist()
df_sample['label_약간'] = df_sample['label_약간'].astype(str)
lst_label = df_sample['label_약간'].values.tolist()

lst_zip = list(map(list,zip(lst_sentences, lst_label)))

print(lst_zip[0])

with open("./little_dataset_test.txt", "wb") as fp:
    pickle.dump(lst_zip, fp)

lst_sentences = df_sample['만_매우'].values.tolist()
df_sample['label_만_매우'] = df_sample['label_만_매우'].astype(str)
lst_label = df_sample['label_만_매우'].values.tolist()

lst_zip = list(map(list,zip(lst_sentences, lst_label)))

print(lst_zip[0])

with open("./man_very_dataset_test.txt", "wb") as fp:
    pickle.dump(lst_zip, fp)
    

lst_sentences = df_sample['만_약간'].values.tolist()
df_sample['label_만_약간'] = df_sample['label_만_약간'].astype(str)
lst_label = df_sample['label_만_약간'].values.tolist()

lst_zip = list(map(list,zip(lst_sentences, lst_label)))

print(lst_zip[0])

with open("./man_little_dataset_test.txt", "wb") as fp:
    pickle.dump(lst_zip, fp)


####
lst_sentences = df_sample['도_매우'].values.tolist()
df_sample['label_도_매우'] = df_sample['label_도_매우'].astype(str)
lst_label = df_sample['label_도_매우'].values.tolist()

lst_zip = list(map(list,zip(lst_sentences, lst_label)))

print(lst_zip[0])

with open("./do_very_dataset_test.txt", "wb") as fp:
    pickle.dump(lst_zip, fp)
    

lst_sentences = df_sample['도_약간'].values.tolist()
df_sample['label_도_약간'] = df_sample['label_도_약간'].astype(str)
lst_label = df_sample['label_도_약간'].values.tolist()

lst_zip = list(map(list,zip(lst_sentences, lst_label)))

print(lst_zip[0])

with open("./do_little_dataset_test.txt", "wb") as fp:
    pickle.dump(lst_zip, fp)

lst_sentences = df_sample['은/는_매우'].values.tolist()
df_sample['label_은/는_매우'] = df_sample['label_은/는_매우'].astype(str)
lst_label = df_sample['label_은/는_매우'].values.tolist()

lst_zip = list(map(list,zip(lst_sentences, lst_label)))

print(lst_zip[0])

with open("./eun_neun_very_dataset_test.txt", "wb") as fp:
    pickle.dump(lst_zip, fp)
    

lst_sentences = df_sample['은/는_약간'].values.tolist()
df_sample['label_은/는_약간'] = df_sample['label_은/는_약간'].astype(str)
lst_label = df_sample['label_은/는_약간'].values.tolist()

lst_zip = list(map(list,zip(lst_sentences, lst_label)))

print(lst_zip[0])

with open("./eun_neun_little_dataset_test.txt", "wb") as fp:
    pickle.dump(lst_zip, fp)