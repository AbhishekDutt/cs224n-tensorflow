---
title: "05. Linguistic Structure: Dependency Parsing"
metaTitle: "This is the title tag of this page"
metaDescription: "This is the meta description"
---

# Lecture plan: Dependency parsing
1. Syntactic Structure: Consistency and Dependency (25 mins)
2. Dependency Grammar and Treebanks (15 mins)
3. Transition-based dependency parsing (15 mins)
4. Neural dependency parsing (15 mins)

In this lecture, two views of linguistic structures were discussed.

1. Constituency = phrase structure grammar = context-free grammars (CFGs): organizes words into nested constituents.
2. Dependency structure: Dependency structure shows which words depend on (modify or are arguments of) which other words.

# Why do we need sentence structure?
- We need to understand sentence structure in order to be able to interpret language correctly
- Humans communicate complex ideas by composing words together into bigger units to convey complex meanings
- We need to know what is connected to what

# Methods of Dependency Parsing
1. Dynamic programming: Eisner (1996) gives a clever algorithm with complexity O(n3), by producing parse items with heads at the ends rather than in the middle
2. Graph algorithms: You create a Minimum Spanning Tree for a sentence. McDonald et al.’s (2005) MSTParser scores dependencies independently using an ML classifier (he uses MIRA, for online learning, but it can be something else)
3. Constraint Satisfaction: Edges are eliminated that don’t satisfy hard constraints. Karlsson (1990), etc.
4. “Transition-based parsing” or “deterministic dependency parsing”: Greedy choice of attachments guided by good machine learning classifiers MaltParser (Nivre et al. 2008). Has proven highly effective.
 
 