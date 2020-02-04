#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## References:

# https://www.cs.bgu.ac.il/~elhadad/nlp16/NLTK-PCFG.html
# http://www.nltk.org/book/ch08-extras.html
# http://www.nltk.org/howto/parse.html


# In[ ]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[ ]:


import nltk
import nltk.corpus


# In[ ]:


nltk.corpus.treebank.fileids()


# In[ ]:


from nltk.corpus import treebank
print(treebank.words('wsj_0007.mrg'))
print(treebank.tagged_words('wsj_0007.mrg'))
print(treebank.parsed_sents('wsj_0007.mrg')[2])


# In[ ]:


# Dealing with parsed sentences {they are nltk.Tree}
treebank.parsed_sents('wsj_0007.mrg')[2].draw()
treebank.parsed_sents('wsj_0007.mrg')[2].productions()


# In[ ]:


# parsed sentences are Trees

from nltk import Tree

# Parse a tree from a string with parentheses.
s = '(S (NP (DT the) (NN cat)) (VP (VBD ate) (NP (DT a) (NN cookie))))'
t = Tree.fromstring(s)
print(t)
t.draw()
print("Display tree properties:")
print(t.label())   # tree's constituent type
print(t[0]) # tree's first child
print(t[1]) # tree's second child
print(t.height())
print(t.leaves())
print(t[1])
print(t[1,1])
print(t[1,1,0])


# In[ ]:


t.productions()


# In[ ]:


# Demonstrate tree modification.
the_cat = t[0]
the_cat.insert(1, Tree.fromstring('(JJ big)'))
print("Tree modification:")
print(t)
t[1,1,1] = Tree.fromstring('(NN cake)')
print(t)


# In[ ]:


# Tree transforms
print("Collapse unary:")
t.collapse_unary()
print(t)
print("Chomsky normal form:")
t.chomsky_normal_form()
print(t)


# In[ ]:


# Demonstrate probabilistic trees.
pt = nltk.tree.ProbabilisticTree('x', ['y', 'z'], prob=0.5)
print("Probabilistic Tree:")
print(pt)
pt.draw()


# In[ ]:


## Grammar tools
import nltk
from nltk import Nonterminal, nonterminals, Production, CFG
nonterminal1 = Nonterminal('NP')
nonterminal2 = Nonterminal('VP')
nonterminal3 = Nonterminal('PP')
print(nonterminal1.symbol())
print(nonterminal2.symbol())
print(nonterminal1==nonterminal2)
S, NP, VP, PP = nonterminals('S, NP, VP, PP') ## use nonterminals to generate a list 
N, V, P, DT = nonterminals('N, V, P, DT')
production1 = Production(S, [NP, VP])
production2 = Production(NP, [DT, NP])
production3 = Production(VP, [V, NP,NP,PP])
print(production1.lhs())
print(production1.rhs())
print(production3 == Production(VP, [V,NP,NP,PP]))


# In[ ]:


nltk.download('large_grammars')


# In[ ]:


### ATIS grammer
import nltk
gram1 = nltk.data.load('grammars/large_grammars/atis.cfg')
gram1
grammar.check_coverage(['a'])


# In[ ]:


print(gram1)


# In[ ]:


## Two kinds of parsers; Normal with CFG and Probabilistic with PCFG. Normal can accept PCFG but won't use probabilities


# In[ ]:


## BottomUpLeftCornerChartParser, LeftCornerChartParser, TopDownChartParser, IncrementalBottomUpChartParser
sent = nltk.data.load('grammars/large_grammars/atis_sentences.txt')
sent = nltk.parse.util.extract_test_sentences(sent)
testingsent=sent[25]
sent=testingsent[0]
parser1 = nltk.parse.EarleyChartParser(gram1)
chart1 = parser1.chart_parse(sent)
print((chart1.num_edges()))
print((len(list(chart1.parses(gram1.start())))))


# In[ ]:


### parser.parse() returns generator
chart1 = parser1.parse(sent)
next(chart1)


# In[ ]:


chart1 = parser1.parse(sent)
print(chart1)
parsed_sents = list(chart1)
print(chart1)


# In[ ]:


print(parsed_sents[0])


# In[ ]:


parsed_sents[0].draw()


# In[ ]:


nltk.parse.chart.demo(5, print_times=False, trace=1, sent='I saw John with a dog', numparses=2)


# In[ ]:


### PCFG
# https://www.nltk.org/api/nltk.parse.html#module-nltk.parse.pchart


# In[ ]:


from nltk.grammar import PCFG, induce_pcfg, toy_pcfg1, toy_pcfg2
# nltk.parse_pcfg not working
grammar = PCFG.fromstring("""
A -> B B [.3] | C B C [.7]
B -> B D [.5] | C [.5]
C -> 'a' [.1] | 'b' [0.9]
D -> 'b' [1.0]
""")
grammar.productions()


# In[ ]:


# get grammer from parsed sentences
from nltk import Nonterminal

productions = []
for fileid in treebank.fileids()[:2]:
    for t in treebank.parsed_sents(fileid):
        productions += t.productions()
grammar = induce_pcfg(Nonterminal('S'), productions)


# In[ ]:


print(grammar)


# In[ ]:


sorted(grammar.productions(lhs=Nonterminal('PP')))[:2]


# In[ ]:


sorted(grammar.productions(lhs=Nonterminal('NNP')))[:2]


# In[ ]:


tokens = "Jack saw Bob with my cookie".split()
grammar = toy_pcfg2
print(grammar)


# In[ ]:


from nltk.parse import pchart
parser = pchart.InsideChartParser(grammar)
for t in parser.parse(tokens):
    print(t)
### try RandomChartParser, UnsortedChartParser, LongestChartParser


# In[ ]:


parser = nltk.parse.EarleyChartParser(grammar)
for t in parser.parse(tokens):
    print(t)


# In[ ]:


### CYK parser gets the most probable parse
from nltk.parse import ViterbiParser

parser = ViterbiParser(grammar)
parser.trace(3)
parsed_sent = list(parser.parse_all(tokens)) # to convert generator to list
parsed_sent[0].draw()
for t in parsed_sent:
    print(t)


# In[ ]:


### CYK parser gets the most probable parse
from nltk.parse import ViterbiParser

parser = ViterbiParser(grammar)
parser.trace(3)
parsed_sent = list(parser.parse(tokens)) # to convert generator to list
parsed_sent[0].draw()
for t in parsed_sent:
    print(t)


# In[ ]:




