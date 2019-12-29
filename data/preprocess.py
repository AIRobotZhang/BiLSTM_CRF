# -*- coding:utf-8 -*-

f_corpus = open('train_corpus.txt', encoding='utf-8')
f_label = open('train_label.txt', encoding='utf-8')
f_w = open('train.tsv', 'w', encoding='utf-8')
for text, label in zip(f_corpus, f_label):
	text = text.strip()
	label = label.strip()
	f_w.write(text+'\t'+label+'\n')
f_corpus.close()
f_label.close()
f_w.close()

f_corpus = open('test_corpus.txt', encoding='utf-8')
f_label = open('test_label.txt', encoding='utf-8')
f_w = open('test.tsv', 'w', encoding='utf-8')
for text, label in zip(f_corpus, f_label):
	text = text.strip()
	label = label.strip()
	f_w.write(text+'\t'+label+'\n')
f_corpus.close()
f_label.close()
f_w.close()