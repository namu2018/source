# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 12:35:18 2018

@author: ktm
"""
#%%
from bs4 import BeautifulSoup
import requests as rq
import lxml


url='https://edition.cnn.com/search/?q=trump&size=10&from=10&page=2'
r=rq.get(url)
print(r.content)
soup = BeautifulSoup(r.content, 'lxml')

soup2 = soup.find('span') #<section class="result news"> 가져오기
# for article in soup2.findAll('dd',attrs={'class':'thumb'}):
print(soup2.text)


#%%

<h3 class="cnn-search__result-headline">
<a href="//www.cnn.com/2018/12/10/politics/epa-wotus/index.html">EPA expected to announces new definition of waters protected under Clean Water Act</a>
                        </h3>
<div class="cnn-search__result-contents">
                        <h3 class="cnn-search__result-headline">
                        <a href="//www.cnn.com/2018/12/10/politics/epa-wotus/index.html">EPA expected to announces new definition of waters protected under Clean Water Act</a>
                        </h3>
                        <div class="cnn-search__result-publish-date">
                            <span class="icon icon--timestamp"></span>
                            <span>Dec 11, 2018</span>
                        </div>
                        <div class="cnn-search__result-body">
                            The Environmental Protection Agency is widely expected to announce plans to change the definition of which waters in the United States are protected under the Clean Water Act on Tuesday -- a change President Donald Trump has been working toward since the beginning of his presidency.  The EPA released a statement saying it would make a major water policy announcement on Tuesday.   The announcement is expected to be a policy shift from former President Barack Obama's 2015 Waters of the United States regulation. Obama's rule changed the definition of which bodies of water the federal government had authority over to include streams and wetlands so that the government could ensure that those waterways remained pollution-free. It altered the definition from the initial one established by the EPA and Army Corps of Engineers in the 1980s. Obama's regulation was created under the Clean Water Act, which regulates discharges of pollutants into waters in the US. Under the Clean Water Act, it's illegal to discharge pollutants from a source into "navigable waters," according to the EPA. The EPA has been working toward a replacement of the regulation since Trump took office. In February 2017, less than two months after his inauguration, Trump signed an executive order asking the EPA and the Department of the Army to review Obama-era water regulations and make sure they are not harming the economy. The EPA's announcement is another step toward Trump fulfilling a campaign promise. At a campaign event at the Economic Club of New York in September 2016, Trump said he would repeal both the Clean Power Plan and the Waters of the United States rule. "I will eliminate all needless and job killing regulations now on the books, and there are plenty of them," Trump said. "This includes eliminating some of our most intrusive regulations like the Waters of the US Rule. It also means scrapping the EPA's so-called Clean Power Plan, which the government itself estimates will cost $7.2 billion a year." Critics, including some in the farming community, have complained that the rule restricted how they could use their land and had a negative economic impact on their business. A few months after Trump's executive order, in July 2017, the EPA issued a proposed action to repeal the 2015 rule and opened a public comment period through August 2018. On February 6, the Trump administration published a rule in the Federal Register that delayed the rule until 2020 and reverted the Waters of the United States interpretation back to the definition applied in the 1980s, according to court documents. Several environmental groups sued the federal government through a district court in South Carolina, arguing that the EPA and Department of the Army's repeal of the regulation violated the Administrative Procedures Act by not having a public comment period. The act governs how federal agencies implement and alter regulations. A federal district court judge sided with environmentalists and issued an injunction nationwide, effectively putting Obama's rule back in place in 22 states, the District of Columbia and US territories, according to the EPA.
                        </div>
                    </div>