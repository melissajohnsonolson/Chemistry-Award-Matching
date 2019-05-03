# Chemistry-Award-Matching
These sets of scripts allow a user to match their own research description to funded chemistry awards and see the publication trends emerging from those awards

The first step in setting up this routine is downloading the award informatin from the National Science Foundation. A copy of data is included in this repository, but it can also be found here:

https://www.nsf.gov/awardsearch/advancedSearch.jsp

It does not necessairly need to be chemistry data, and can come from any division or directorate, but be sure to alter the pgms parameter in the model training script as it currently filters out all programs except for the core chemistry research programs. 

The publication data used later on can be generated with the crossref paper data script. It reads in the awards numbers from the data file above and returns publication data tagged with each award number.

Once this data has all been collected, the Doc2Vec model can be trained with the doc2vec abstract training script. It is wrapped up as a function, but each step is broken down below:

Import the necessary libraries and set the directory to the location of the abstract data. Load the csv into the function.
<body>
  <div tabindex="-1" id="notebook" class="border-box-sizing">
    <div class="container" id="notebook-container">

<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[1]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">import</span> <span class="nn">os</span>
<span class="kn">import</span> <span class="nn">string</span>
<span class="kn">import</span> <span class="nn">pandas</span> <span class="k">as</span> <span class="nn">pd</span>
<span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
<span class="kn">from</span> <span class="nn">nltk.corpus</span> <span class="k">import</span> <span class="n">stopwords</span> 
<span class="kn">from</span> <span class="nn">nltk.stem.wordnet</span> <span class="k">import</span> <span class="n">WordNetLemmatizer</span>
<span class="kn">import</span> <span class="nn">re</span>
<span class="kn">import</span> <span class="nn">datetime</span>
<span class="kn">import</span> <span class="nn">gensim</span>
<span class="kn">from</span> <span class="nn">gensim.models.doc2vec</span> <span class="k">import</span> <span class="n">Doc2Vec</span><span class="p">,</span> <span class="n">TaggedDocument</span>
<span class="kn">import</span> <span class="nn">matplotlib.pyplot</span> <span class="k">as</span> <span class="nn">plt</span>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[2]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">awds</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_csv</span><span class="p">(</span><span class="s1">&#39;NSF CHE 2012.csv&#39;</span><span class="p">,</span> <span class="n">encoding</span><span class="o">=</span><span class="s1">&#39;latin-1&#39;</span><span class="p">)</span>
</span>
<span class="n">awds</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
</pre></div>

 
Here you can see the format of the awards data



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AwardNumber</th>
      <th>Title</th>
      <th>NSFOrganization</th>
      <th>Program(s)</th>
      <th>StartDate</th>
      <th>LastAmendmentDate</th>
      <th>PrincipalInvestigator</th>
      <th>State</th>
      <th>Organization</th>
      <th>AwardInstrument</th>
      <th>...</th>
      <th>OrganizationStreet</th>
      <th>OrganizationCity</th>
      <th>OrganizationState</th>
      <th>OrganizationZip</th>
      <th>OrganizationPhone</th>
      <th>NSFDirectorate</th>
      <th>ProgramElementCode(s)</th>
      <th>ProgramReferenceCode(s)</th>
      <th>ARRAAmount</th>
      <th>Abstract</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1265397</td>
      <td>GOALI: New Techniques for Solid-State NMR Stud...</td>
      <td>CHE</td>
      <td>GRANT OPP FOR ACAD LIA W/INDUS, ANALYTICAL SEP...</td>
      <td>2010</td>
      <td>07/19/2013</td>
      <td>Eric Munson</td>
      <td>KY</td>
      <td>University of Kentucky Research Foundation</td>
      <td>Continuing grant</td>
      <td>...</td>
      <td>109 Kinkead Hall</td>
      <td>Lexington</td>
      <td>KY</td>
      <td>405260001</td>
      <td>8592579420</td>
      <td>MPS</td>
      <td>1504, 1974</td>
      <td>0000, 1504, 9150, 9161, OTHR</td>
      <td>$0.00</td>
      <td>Professor Eric Munson of the University of Kan...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1202641</td>
      <td>Data Portal Enabling New Protein Structure Col...</td>
      <td>CHE</td>
      <td>ANALYTICAL SEPARATIONS &amp; MEAS.</td>
      <td>2010</td>
      <td>01/11/2012</td>
      <td>Daniele Fabris</td>
      <td>NY</td>
      <td>SUNY at Albany</td>
      <td>Continuing grant</td>
      <td>...</td>
      <td>1400 WASHINGTON AVE MSC 100A</td>
      <td>Albany</td>
      <td>NY</td>
      <td>122220100</td>
      <td>5184374974</td>
      <td>MPS</td>
      <td>1974</td>
      <td>0000, OTHR</td>
      <td>$0.00</td>
      <td>Professor Daniele Fabris of the University of ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1241424</td>
      <td>REU Site: Research Experiences for Undergradua...</td>
      <td>CHE</td>
      <td>UNDERGRADUATE PROGRAMS IN CHEM</td>
      <td>2011</td>
      <td>05/30/2012</td>
      <td>Joanne Romagni</td>
      <td>IL</td>
      <td>DePaul University</td>
      <td>Continuing grant</td>
      <td>...</td>
      <td>1 East Jackson Boulevard</td>
      <td>Chicago</td>
      <td>IL</td>
      <td>606042287</td>
      <td>3123627595</td>
      <td>MPS</td>
      <td>1986</td>
      <td>9178, 9250, SMET</td>
      <td>$0.00</td>
      <td>This program establishes a new summer Internat...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1216129</td>
      <td>CAREER: Investigation of biological signaling ...</td>
      <td>CHE</td>
      <td>Chemistry of Life Processes</td>
      <td>2011</td>
      <td>02/27/2017</td>
      <td>Shawn Burdette</td>
      <td>MA</td>
      <td>Worcester Polytechnic Institute</td>
      <td>Continuing grant</td>
      <td>...</td>
      <td>100 INSTITUTE RD</td>
      <td>WORCESTER</td>
      <td>MA</td>
      <td>16092247</td>
      <td>5088315000</td>
      <td>MPS</td>
      <td>6883</td>
      <td>1045, 1187, 9183</td>
      <td>$0.00</td>
      <td>This research award in the Chemistry of Life P...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1230357</td>
      <td>CAREER: Generation and Fate of Distonic Radica...</td>
      <td>CHE</td>
      <td>UNIMOLECULAR PROCESSES, Structure,Dynamics &amp;Me...</td>
      <td>2011</td>
      <td>08/15/2012</td>
      <td>Michelle Claville</td>
      <td>VA</td>
      <td>Hampton University</td>
      <td>Standard Grant</td>
      <td>...</td>
      <td>100 E. Queen Street</td>
      <td>Hampton</td>
      <td>VA</td>
      <td>236680108</td>
      <td>7577275363</td>
      <td>MPS</td>
      <td>1942, 6879, 9150</td>
      <td>0000, 1045, 1187, 1982, 6879, 9150, 9161, AMPP...</td>
      <td>$0.00</td>
      <td>Abstract&lt;br/&gt;&lt;br/&gt;&lt;br/&gt;&lt;br/&gt;Proposal: 0847742 ...</td>
    </tr>
  </tbody>
</table>

Here we filter by program elements to ensure only the core research programs are considered. the pgms parameter can be changed for any set of research programs one may want to examine (for example, to include Major Research Instrumetation or Research Exeperiences of Undergradutes). Here we also generate lists of 'stop' words and punctuation to remove and instantiate a lemmatizer to help clean up the abstract text.

<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[4]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">#First we need to filter the data by program code. Some grants have multiple program</span>
<span class="c1">#codes, so we first filter through to determine which cells contain the program code</span>
<span class="c1">#then we replace the exisiting program code(s) with the provided one. This ensures there</span>
<span class="c1">#is only one code per award.</span>

<span class="n">pgms</span> <span class="o">=</span> <span class="p">[</span><span class="s1">&#39;6878&#39;</span><span class="p">,</span> <span class="s1">&#39;6880&#39;</span><span class="p">,</span> <span class="s1">&#39;6882&#39;</span><span class="p">,</span> <span class="s1">&#39;6883&#39;</span><span class="p">,</span> <span class="s1">&#39;6884&#39;</span><span class="p">,</span> <span class="s1">&#39;6885&#39;</span><span class="p">,</span> <span class="s1">&#39;9101&#39;</span><span class="p">,</span> <span class="s1">&#39;9102&#39;</span><span class="p">,</span> <span class="s1">&#39;6881&#39;</span><span class="p">]</span>

<span class="n">awds</span> <span class="o">=</span> <span class="n">awds</span><span class="p">[</span><span class="n">awds</span><span class="p">[</span><span class="s1">&#39;ProgramElementCode(s)&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">str</span><span class="o">.</span><span class="n">contains</span><span class="p">(</span><span class="s1">&#39;|&#39;</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">pgms</span><span class="p">))]</span>
<span class="k">for</span> <span class="n">x</span> <span class="ow">in</span> <span class="n">pgms</span><span class="p">:</span>
    <span class="n">awds</span><span class="p">[</span><span class="s1">&#39;ProgramElementCode(s)&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">where</span><span class="p">(</span><span class="n">awds</span><span class="p">[</span><span class="s1">&#39;ProgramElementCode(s)&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">str</span><span class="o">.</span><span class="n">contains</span><span class="p">(</span><span class="n">x</span><span class="p">),</span> <span class="n">x</span><span class="p">,</span> <span class="n">awds</span><span class="p">[</span><span class="s1">&#39;ProgramElementCode(s)&#39;</span><span class="p">]</span> <span class="p">)</span>
   
<span class="n">abstracts</span> <span class="o">=</span> <span class="n">awds</span><span class="p">[[</span><span class="s1">&#39;AwardNumber&#39;</span><span class="p">,</span><span class="s1">&#39;Abstract&#39;</span><span class="p">]]</span><span class="o">.</span><span class="n">copy</span><span class="p">()</span>
<span class="c1">#This is a pretty clean data set, but there are some empty entries, so we</span>
<span class="c1">#filter them out here</span>
<span class="n">abstracts</span> <span class="o">=</span> <span class="n">abstracts</span><span class="o">.</span><span class="n">dropna</span><span class="p">()</span>
<span class="c1">#Here we start building our dictinary and creating the cleaned up corpus.</span>
<span class="c1">#We start by  removing stop words, punctuation, and lemmatizing</span>
<span class="c1">#the abstract text</span>
<span class="n">stop</span>    <span class="o">=</span> <span class="nb">set</span><span class="p">(</span><span class="n">stopwords</span><span class="o">.</span><span class="n">words</span><span class="p">(</span><span class="s1">&#39;english&#39;</span><span class="p">))</span>
<span class="n">exclude</span> <span class="o">=</span> <span class="nb">set</span><span class="p">(</span><span class="n">string</span><span class="o">.</span><span class="n">punctuation</span><span class="p">)</span> 
<span class="n">lemma</span>   <span class="o">=</span> <span class="n">WordNetLemmatizer</span><span class="p">()</span>
<span class="n">boiler_plate</span> <span class="o">=</span> <span class="s1">&#39;This award reflects NSF&#39;&#39;s statutory mission and has been deemed worthy of support through evaluation using the Foundation&#39;&#39;s intellectual merit and broader impacts review criteria&#39;</span> 
</pre></div>

Now everything is wrapped up into a function and we can pass each each abstract through it. This function also tokenizes each abstract, that is breaks it into a list of individual words.

<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[5]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">#This function puts it all together and we can pass the abstracts through</span>
<span class="k">def</span> <span class="nf">word_mod</span><span class="p">(</span><span class="n">doc</span><span class="p">):</span>
    <span class="n">doc</span> <span class="o">=</span> <span class="n">re</span><span class="o">.</span><span class="n">sub</span><span class="p">(</span><span class="s1">&#39;&lt;.*?&gt;&#39;</span><span class="p">,</span> <span class="s1">&#39; &#39;</span><span class="p">,</span> <span class="n">doc</span><span class="p">)</span>
    <span class="n">doc</span> <span class="o">=</span> <span class="n">re</span><span class="o">.</span><span class="n">sub</span><span class="p">(</span><span class="n">boiler_plate</span><span class="p">,</span> <span class="s1">&#39;&#39;</span><span class="p">,</span> <span class="n">doc</span><span class="p">)</span>
    <span class="n">punct_free</span>  <span class="o">=</span> <span class="s1">&#39;&#39;</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">ch</span> <span class="k">for</span> <span class="n">ch</span> <span class="ow">in</span> <span class="n">doc</span> <span class="k">if</span> <span class="n">ch</span> <span class="ow">not</span> <span class="ow">in</span> <span class="n">exclude</span><span class="p">)</span>
    <span class="n">words</span>   <span class="o">=</span> <span class="n">punct_free</span><span class="o">.</span><span class="n">lower</span><span class="p">()</span><span class="o">.</span><span class="n">split</span><span class="p">()</span>
    <span class="n">stop_free</span>  <span class="o">=</span> <span class="s2">&quot; &quot;</span><span class="o">.</span><span class="n">join</span><span class="p">([</span><span class="n">i</span> <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="n">words</span> <span class="k">if</span> <span class="n">i</span> <span class="ow">not</span> <span class="ow">in</span> <span class="n">stop</span><span class="p">])</span>
    <span class="n">lemm</span> <span class="o">=</span> <span class="s2">&quot; &quot;</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">lemma</span><span class="o">.</span><span class="n">lemmatize</span><span class="p">(</span><span class="n">word</span><span class="p">)</span> <span class="k">for</span> <span class="n">word</span> <span class="ow">in</span> <span class="n">stop_free</span><span class="o">.</span><span class="n">split</span><span class="p">())</span>
    <span class="n">word_list</span> <span class="o">=</span> <span class="n">lemm</span><span class="o">.</span><span class="n">split</span><span class="p">()</span>
    <span class="c1"># only take words which are greater than 2 characters</span>
    <span class="n">cleaned</span> <span class="o">=</span> <span class="p">[</span><span class="n">word</span> <span class="k">for</span> <span class="n">word</span> <span class="ow">in</span> <span class="n">word_list</span> <span class="k">if</span> <span class="nb">len</span><span class="p">(</span><span class="n">word</span><span class="p">)</span> <span class="o">&gt;</span> <span class="mi">2</span><span class="p">]</span>
    <span class="k">return</span> <span class="n">cleaned</span>
    
<span class="n">abstracts</span><span class="p">[</span><span class="s1">&#39;clean_abstracts&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="p">[</span><span class="n">word_mod</span><span class="p">(</span><span class="n">doc</span><span class="p">)</span> <span class="k">for</span> <span class="n">doc</span> <span class="ow">in</span> <span class="n">abstracts</span><span class="p">[</span><span class="s1">&#39;Abstract&#39;</span><span class="p">]]</span> 
<span class="nb">print</span><span class="p">(</span><span class="n">abstracts</span><span class="p">[</span><span class="s1">&#39;clean_abstracts&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">iloc</span><span class="p">[</span><span class="mi">2</span><span class="p">])</span>
</pre>
Here is was the abstract now looks like - tokenized and cleaned
<pre>[&#39;project&#39;, &#39;funded&#39;, &#39;chemical&#39;, &#39;synthesis&#39;, &#39;program&#39;, &#39;chemistry&#39;, &#39;division&#39;, &#39;professor&#39;, &#39;tyler&#39;, &#39;mcquade&#39;, &#39;department&#39;, &#39;chemistry&#39;, &#39;biochemistry&#39;, &#39;florida&#39;, &#39;state&#39;, &#39;university&#39;, &#39;develop&#39;, &#39;new&#39;, &#39;synthetic&#39;, &#39;method&#39;, &#39;creating&#39;, &#39;stereoconvergent&#39;, &#39;allylic&#39;, &#39;substitution&#39;, &#39;providing&#39;, &#39;stereoisomer&#39;, &#39;trans&#39;, &#39;substrate&#39;, &#39;catalystcontrolled&#39;, &#39;reactivity&#39;, &#39;neighboring&#39;, &#39;stereocenters&#39;, &#39;limited&#39;, &#39;impact&#39;, &#39;diastereoselectivity&#39;, &#39;regiocontrolled&#39;, &#39;catalytic&#39;, &#39;hydroborations&#39;, &#39;rare&#39;, &#39;metal&#39;, &#39;rhodium&#39;, &#39;iridium&#39;, &#39;become&#39;, &#39;expensive&#39;, &#39;critically&#39;, &#39;important&#39;, &#39;increase&#39;, &#39;range&#39;, &#39;reaction&#39;, &#39;performed&#39;, &#39;inexpensive&#39;, &#39;metal&#39;, &#39;copper&#39;, &#39;resulting&#39;, &#39;method&#39;, &#39;could&#39;, &#39;efficient&#39;, &#39;selective&#39;, &#39;prior&#39;, &#39;reaction&#39;, &#39;could&#39;, &#39;enable&#39;, &#39;new&#39;, &#39;strategy&#39;, &#39;construct&#39;, &#39;valuable&#39;, &#39;complex&#39;, &#39;molecule&#39;, &#39;expensive&#39;, &#39;metal&#39;, &#39;providing&#39;, &#39;basic&#39;, &#39;science&#39;, &#39;necessary&#39;, &#39;improve&#39;, &#39;sustainability&#39;, &#39;proposed&#39;, &#39;work&#39;, &#39;predicted&#39;, &#39;result&#39;, &#39;thorough&#39;, &#39;understanding&#39;, &#39;catalyst&#39;, &#39;provide&#39;, &#39;greater&#39;, &#39;chemical&#39;, &#39;control&#39;, &#39;also&#39;, &#39;creation&#39;, &#39;promising&#39;, &#39;new&#39;, &#39;synthetic&#39;, &#39;method&#39;, &#39;successful&#39;, &#39;result&#39;, &#39;work&#39;, &#39;positively&#39;, &#39;impact&#39;, &#39;pharmaceutical&#39;, &#39;agrochemical&#39;, &#39;specialty&#39;, &#39;chemical&#39;, &#39;industry&#39;, &#39;addition&#39;, &#39;project&#39;, &#39;provide&#39;, &#39;excellent&#39;, &#39;training&#39;, &#39;student&#39;, &#39;undergraduate&#39;, &#39;postdoctoral&#39;, &#39;including&#39;, &#39;group&#39;, &#39;historically&#39;, &#39;underrepresented&#39;, &#39;science&#39;, &#39;student&#39;, &#39;become&#39;, &#39;vanguard&#39;, &#39;leading&#39;, &#39;way&#39;, &#39;future&#39;, &#39;environmentallyfriendly&#39;, &#39;internationallycompetitive&#39;, &#39;organic&#39;, &#39;chemistry&#39;]
</pre>
Now we can start training the model. First every abstract is tagged with it's award id, then the corpus of tagged abstracts is passed through the doc2vec model. The parameters of this model can be changed, please see the doc2vec documentation from gensim to do so.
<div class="input">
<div class="prompt input_prompt">In&nbsp;[7]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">#Now the model can be trained. First we create a tagged set - the cleaned</span>
<span class="c1">#abstracts tagged with their award number. The model is instantiated, trained</span>
<span class="c1">#and saved</span>
<span class="n">train_corpus</span> <span class="o">=</span> <span class="p">[</span><span class="n">gensim</span><span class="o">.</span><span class="n">models</span><span class="o">.</span><span class="n">doc2vec</span><span class="o">.</span><span class="n">TaggedDocument</span><span class="p">(</span><span class="n">abstracts</span><span class="p">[</span><span class="s1">&#39;clean_abstracts&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">iloc</span><span class="p">[</span><span class="n">i</span><span class="p">],</span> <span class="p">[</span><span class="nb">str</span><span class="p">(</span><span class="n">abstracts</span><span class="p">[</span><span class="s1">&#39;AwardNumber&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">iloc</span><span class="p">[</span><span class="n">i</span><span class="p">])])</span> <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">abstracts</span><span class="p">))]</span>    
<span class="n">model</span> <span class="o">=</span> <span class="n">gensim</span><span class="o">.</span><span class="n">models</span><span class="o">.</span><span class="n">doc2vec</span><span class="o">.</span><span class="n">Doc2Vec</span><span class="p">(</span><span class="n">vector_size</span><span class="o">=</span><span class="mi">50</span><span class="p">,</span> <span class="n">min_count</span><span class="o">=</span><span class="mi">2</span><span class="p">,</span> <span class="n">epochs</span><span class="o">=</span><span class="mi">100</span><span class="p">)</span>    
<span class="n">model</span><span class="o">.</span><span class="n">build_vocab</span><span class="p">(</span><span class="n">train_corpus</span><span class="p">)</span>
<span class="n">model</span><span class="o">.</span><span class="n">train</span><span class="p">(</span><span class="n">train_corpus</span><span class="p">,</span> <span class="n">total_examples</span><span class="o">=</span><span class="n">model</span><span class="o">.</span><span class="n">corpus_count</span><span class="p">,</span> <span class="n">epochs</span><span class="o">=</span><span class="n">model</span><span class="o">.</span><span class="n">epochs</span><span class="p">)</span>   
<span class="n">model</span><span class="o">.</span><span class="n">save</span><span class="p">(</span><span class="s1">&#39;doc2vec_abstracts&#39;</span><span class="p">)</span>
</pre>
 We can run a 'sanity check' on our model such that when an abstract the model is trained on is passed through the model, the model correctly identifies the same abstract as the most similar.
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[8]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">model</span> <span class="o">=</span> <span class="n">Doc2Vec</span><span class="o">.</span><span class="n">load</span><span class="p">(</span><span class="s1">&#39;doc2vec_abstracts&#39;</span><span class="p">)</span>  
<span class="c1">#A sanity check for our models perfromace is to pass the abstracts back</span>
<span class="c1">#the model and see if the model-determined most similar award number</span>
<span class="c1">#corresponds to the abstract we passed.</span>
<span class="n">rank</span> <span class="o">=</span> <span class="mi">0</span>
<span class="n">sim</span> <span class="o">=</span><span class="p">[]</span>
<span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">abstracts</span><span class="p">)):</span>
    <span class="n">inferred_vector</span> <span class="o">=</span> <span class="n">model</span><span class="o">.</span><span class="n">infer_vector</span><span class="p">(</span><span class="n">abstracts</span><span class="p">[</span><span class="s1">&#39;clean_abstracts&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">iloc</span><span class="p">[</span><span class="n">i</span><span class="p">])</span>
    <span class="n">sims</span> <span class="o">=</span> <span class="n">model</span><span class="o">.</span><span class="n">docvecs</span><span class="o">.</span><span class="n">most_similar</span><span class="p">([</span><span class="n">inferred_vector</span><span class="p">],</span> <span class="n">topn</span><span class="o">=</span><span class="nb">len</span><span class="p">(</span><span class="n">model</span><span class="o">.</span><span class="n">docvecs</span><span class="p">))</span>
    <span class="k">if</span> <span class="n">abstracts</span><span class="p">[</span><span class="s1">&#39;AwardNumber&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">iloc</span><span class="p">[</span><span class="n">i</span><span class="p">]</span> <span class="o">!=</span> <span class="nb">int</span><span class="p">(</span><span class="n">sims</span><span class="p">[</span><span class="mi">0</span><span class="p">][</span><span class="mi">0</span><span class="p">]):</span>
        <span class="n">rank</span> <span class="o">=</span> <span class="n">rank</span> <span class="o">+</span><span class="mi">1</span>
    <span class="k">else</span><span class="p">:</span>
        <span class="n">sim</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">sims</span><span class="p">[</span><span class="mi">0</span><span class="p">][</span><span class="mi">1</span><span class="p">])</span>
<span class="n">simscore</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">Series</span><span class="p">(</span><span class="n">sim</span><span class="p">)</span><span class="o">.</span><span class="n">mean</span><span class="p">()</span>
<span class="n">percent</span> <span class="o">=</span> <span class="p">(</span><span class="mi">1</span><span class="o">-</span><span class="n">rank</span><span class="o">/</span><span class="nb">len</span><span class="p">(</span><span class="n">train_corpus</span><span class="p">))</span><span class="o">*</span><span class="mi">100</span>
<span class="nb">print</span><span class="p">(</span><span class="s1">&#39;The model correctly matched the training set </span><span class="si">{0}% o</span><span class="s1">f the time with an average similiarity of </span><span class="si">{1}</span><span class="s1">&#39;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="nb">round</span><span class="p">(</span><span class="n">percent</span><span class="p">,</span><span class="mi">1</span><span class="p">)</span> <span class="p">,</span><span class="nb">round</span><span class="p">(</span><span class="n">simscore</span><span class="p">,</span><span class="mi">3</span><span class="p">)))</span>
    
</pre>

This model correctly matches the abstracts 92% of the time with reasonably high similarity scores.

<div class="output_subarea output_stream output_stdout output_text">
<pre>The model correctly matched the training set 92.1% of the time with an average similiarity of 0.875
</pre>
</div>

Now we can start running research descriptions through the research matching script. The user inputs their own block of text as a string. Below is an example from a published paper's abstract:

<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[9]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">#Now we can test the model.  Here is a sample abstract from Review Toward Infinitely Recyclable Plastics Derived from</span>
<span class="c1"># Renewable Cyclic Esters in the Journal Chem.</span>

<span class="n">new_text</span> <span class="o">=</span> <span class="s1">&#39;The development of biorenewable and chemically recyclable plastics holds real potential to not only preserve natural resources but also solve the end-of-life issue of plastic waste. However, materializing such potential and ultimately establishing a circular plastics economy requires that three challenges be met: energy cost, depolymerization selectivity, and depolymerizability and performance tradeoffs. Recent advances made in this field, especially the discovery of infinitely recyclable plastics, have yielded feasible solutions and design principles. Future directions will focus on designing monomer and polymer structures that deliver properties and performances for tailored application needs while maintaining complete recyclability and catalyst structures and integrated processes with high (de)polymerization activity, selectivity, and efficiency, ultimately solving the severe worldwide environmental problems created by non-recyclable plastics production and disposal.&#39;</span>
<span class="nb">print</span><span class="p">(</span><span class="n">new_text</span><span class="p">)</span>
</pre>
<pre>The development of biorenewable and chemically recyclable plastics holds real potential to not only preserve natural resources but also solve the end-of-life issue of plastic waste. However, materializing such potential and ultimately establishing a circular plastics economy requires that three challenges be met: energy cost, depolymerization selectivity, and depolymerizability and performance tradeoffs. Recent advances made in this field, especially the discovery of infinitely recyclable plastics, have yielded feasible solutions and design principles. Future directions will focus on designing monomer and polymer structures that deliver properties and performances for tailored application needs while maintaining complete recyclability and catalyst structures and integrated processes with high (de)polymerization activity, selectivity, and efficiency, ultimately solving the severe worldwide environmental problems created by non-recyclable plastics production and disposal.
</pre>
Now the text is run the same function as before to clean and tokenize it.  It is run through the model and the first out put is the two most simlar award numbers and the abstract for the most similiar awrad.
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[10]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">#The text is tokenized and passed through the model. The similarity of the text is checked against all of the</span>
<span class="c1">#training text</span>
<span class="n">new_text_clean</span> <span class="o">=</span> <span class="n">model</span><span class="o">.</span><span class="n">infer_vector</span><span class="p">(</span><span class="n">word_mod</span><span class="p">(</span><span class="n">new_text</span><span class="p">))</span>
<span class="n">sims</span> <span class="o">=</span> <span class="n">model</span><span class="o">.</span><span class="n">docvecs</span><span class="o">.</span><span class="n">most_similar</span><span class="p">([</span><span class="n">new_text_clean</span><span class="p">],</span> <span class="n">topn</span><span class="o">=</span><span class="nb">len</span><span class="p">(</span><span class="n">model</span><span class="o">.</span><span class="n">docvecs</span><span class="p">))</span>
<span class="n">sim1</span> <span class="o">=</span> <span class="n">sims</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span>
<span class="n">sim2</span> <span class="o">=</span> <span class="n">sims</span><span class="p">[</span><span class="mi">1</span><span class="p">]</span>
<span class="nb">print</span><span class="p">(</span><span class="s1">&#39;The most similar award numbers are </span><span class="si">{0}</span><span class="s1"> and </span><span class="si">{1}</span><span class="s1">, with similarity scores of </span><span class="si">{2}</span><span class="s1"> and </span><span class="si">{3}</span><span class="s1">.&#39;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">sim1</span><span class="p">[</span><span class="mi">0</span><span class="p">],</span> <span class="n">sim2</span><span class="p">[</span><span class="mi">0</span><span class="p">],</span> <span class="nb">round</span><span class="p">(</span><span class="n">sim1</span><span class="p">[</span><span class="mi">1</span><span class="p">],</span><span class="mi">3</span><span class="p">),</span> <span class="nb">round</span><span class="p">(</span><span class="n">sim2</span><span class="p">[</span><span class="mi">1</span><span class="p">],</span><span class="mi">3</span><span class="p">)))</span>
<span class="nb">print</span><span class="p">(</span><span class="s1">&#39;The most similar award abstract is:&#39;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">awds</span><span class="p">[</span><span class="n">awds</span><span class="p">[</span><span class="s1">&#39;AwardNumber&#39;</span><span class="p">]</span><span class="o">==</span><span class="nb">int</span><span class="p">(</span><span class="n">sim1</span><span class="p">[</span><span class="mi">0</span><span class="p">])][</span><span class="s1">&#39;Abstract&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">values</span><span class="p">)</span>

<span class="c1">#If the text has similarity score of 0.5 or greater to an award abstract, we keep it for further plots</span>
<span class="n">sims</span> <span class="o">=</span> <span class="p">[</span><span class="n">sims</span><span class="p">[</span><span class="n">i</span><span class="p">][</span><span class="mi">0</span><span class="p">]</span> <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">sims</span><span class="p">))</span> <span class="k">if</span> <span class="n">sims</span><span class="p">[</span><span class="n">i</span><span class="p">][</span><span class="mi">1</span><span class="p">]</span><span class="o">&gt;</span><span class="mf">0.5</span><span class="p">]</span>
</pre>
<div class="output_subarea output_stream output_stdout output_text">
<pre>The most similar award numbers are 1413033 and 1610311, with similarity scores of 0.625 and 0.613.
The most similar award abstract is:
[&#39;The research group of Dr. Wenjun Du at the Central Michigan University develops new methodologies for preparing a new class of polymers by linking sugar units together and investigates the conditions for controlling the degradation of these polymers.  Sugar-based polymers are potentially compatible with biological systems and useful in biomedical applications.  The degradation of these polymers under controlled conditions to regenerate the monomers allows recycling of the polymer products and, therefore, is environmentally beneficial.  Outreach and educational activities of this project include developing an online course on sustainable polymer for high school teachers, partnering with the Central Michigan Science/Mathematics/Technology Center in organizing outreach activities for K-12 students and teachers, and mentoring graduate and undergraduate students in research.  &lt;br/&gt;&lt;br/&gt;Under the support of Macromolecular, Supramolecular and Nanochemistry Program of NSF, Dr. Wenjun Du aims to develop synthetic methodogies for preparing degradable polymers by connecting sugar units together via orthoester linkages, thereby circumventing the notoriously challenging syntheses of O-glycosyl linkages.  His research group studies the mechanism, scope and limitation of the polymerization process, explores the synthesis of high molecular weight sugar-based polymers by exploiting the reverse anomeric effect, and investigates the degradation of the sugar poly(orthoesters).&#39;]
</pre>

Next the publication data is loaded and formmatted as shown below.
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[11]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">#The publication data is loaded and formatted. We also want to know the citations per year a paper is garnering</span>
<span class="c1">#a sort-of measure of the impact</span>
<span class="n">papers</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_csv</span><span class="p">(</span><span class="s1">&#39;che_paper_data.csv&#39;</span><span class="p">)</span>
<span class="n">papers</span><span class="p">[</span><span class="s1">&#39;year&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">to_datetime</span><span class="p">(</span><span class="n">papers</span><span class="p">[</span><span class="s1">&#39;year&#39;</span><span class="p">])</span>
<span class="n">papers</span><span class="p">[</span><span class="s1">&#39;citations per year&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="n">papers</span><span class="p">[</span><span class="s1">&#39;citations&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">divide</span><span class="p">(</span>
        <span class="p">[((</span><span class="n">datetime</span><span class="o">.</span><span class="n">datetime</span><span class="o">.</span><span class="n">today</span><span class="p">()</span><span class="o">-</span><span class="n">x</span><span class="p">)</span><span class="o">.</span><span class="n">days</span><span class="p">)</span><span class="o">/</span><span class="mf">365.2422</span> <span class="k">for</span> <span class="n">x</span> <span class="ow">in</span> <span class="n">papers</span><span class="p">[</span><span class="s1">&#39;year&#39;</span><span class="p">]])</span>  
<span class="n">papers</span><span class="p">[</span><span class="s1">&#39;year&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="n">papers</span><span class="p">[</span><span class="s1">&#39;year&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">apply</span><span class="p">(</span><span class="k">lambda</span> <span class="n">x</span><span class="p">:</span><span class="n">x</span><span class="o">.</span><span class="n">year</span><span class="p">)</span>
<span class="n">papers</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
</pre>
    
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>award number</th>
      <th>title</th>
      <th>year</th>
      <th>citations</th>
      <th>type</th>
      <th>publication</th>
      <th>DOI</th>
      <th>funders</th>
      <th>authors</th>
      <th>citations per year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1152845</td>
      <td>Self-consistent phonons revisited. II. A gener...</td>
      <td>2013</td>
      <td>17</td>
      <td>journal-article</td>
      <td>The Journal of Chemical Physics</td>
      <td>10.1063/1.4788977</td>
      <td>[{'DOI': '10.13039/100000001', 'name': 'Nation...</td>
      <td>[{'given': 'Sandra E.', 'family': 'Brown', 'se...</td>
      <td>2.722103</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1152845</td>
      <td>Mapping the phase diagram for neon to a quantu...</td>
      <td>2013</td>
      <td>9</td>
      <td>journal-article</td>
      <td>The Journal of Chemical Physics</td>
      <td>10.1063/1.4796144</td>
      <td>[{'DOI': '10.13039/100000001', 'name': 'Nation...</td>
      <td>[{'given': 'Ionuţ', 'family': 'Georgescu', 'se...</td>
      <td>1.480045</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1152845</td>
      <td>Filter diagonalization method for processing P...</td>
      <td>2013</td>
      <td>18</td>
      <td>journal-article</td>
      <td>Journal of Magnetic Resonance</td>
      <td>10.1016/j.jmr.2013.06.014</td>
      <td>[{'DOI': '10.13039/100000001', 'name': 'Nation...</td>
      <td>[{'given': 'Beau R.', 'family': 'Martini', 'se...</td>
      <td>3.082213</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1152845</td>
      <td>The filter diagonalization method and its asse...</td>
      <td>2014</td>
      <td>5</td>
      <td>journal-article</td>
      <td>International Journal of Mass Spectrometry</td>
      <td>10.1016/j.ijms.2014.08.010</td>
      <td>[{'DOI': '10.13039/100000001', 'name': 'Nation...</td>
      <td>[{'given': 'Beau R.', 'family': 'Martini', 'se...</td>
      <td>1.066089</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1152845</td>
      <td>Assessing the Performance of the Diffusion Mon...</td>
      <td>2015</td>
      <td>10</td>
      <td>journal-article</td>
      <td>The Journal of Physical Chemistry A</td>
      <td>10.1021/acs.jpca.5b02511</td>
      <td>[{'DOI': '10.13039/100000165', 'name': 'Divisi...</td>
      <td>[{'given': 'Joel D.', 'family': 'Mallory', 'se...</td>
      <td>2.536404</td>
    </tr>
  </tbody>
</table>
Now the awards and publication data are matched up based on the similar awards found from the model. And we start making plots!
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[12]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">#now we pull the relavent award and paper data based on the similarity scores</span>
<span class="n">sim_awards</span> <span class="o">=</span> <span class="n">awds</span><span class="p">[</span><span class="n">awds</span><span class="p">[</span><span class="s1">&#39;AwardNumber&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">isin</span><span class="p">(</span><span class="n">sims</span><span class="p">)]</span><span class="o">.</span><span class="n">copy</span><span class="p">()</span>
<span class="n">sim_papers</span> <span class="o">=</span> <span class="n">papers</span><span class="p">[</span><span class="n">papers</span><span class="p">[</span><span class="s1">&#39;award number&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">isin</span><span class="p">(</span><span class="n">sims</span><span class="p">)]</span><span class="o">.</span><span class="n">copy</span><span class="p">()</span>
</pre>
  
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[13]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">#start making plots!</span>
<span class="n">plt</span><span class="o">.</span><span class="n">style</span><span class="o">.</span><span class="n">use</span><span class="p">(</span><span class="s1">&#39;ggplot&#39;</span><span class="p">)</span>
<span class="n">fig1</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">()</span>
<span class="n">sim_awards</span><span class="o">.</span><span class="n">groupby</span><span class="p">(</span><span class="s1">&#39;StartDate&#39;</span><span class="p">)[</span><span class="s1">&#39;AwardNumber&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">count</span><span class="p">()</span><span class="o">.</span><span class="n">plot</span><span class="o">.</span><span class="n">bar</span><span class="p">(</span><span class="n">rot</span> <span class="o">=</span> <span class="mi">0</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s1">&#39;Awards per Year Similar to Text&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="s1">&#39;Number of Awards&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">xlabel</span><span class="p">(</span><span class="s1">&#39;Year of Award&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">tight_layout</span><span class="p">()</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre>
  https://github.com/melissajohnsonolson/Chemistry-Award-Matching/blob/master/Figure_1.png

<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[14]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">fig2</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">()</span>
<span class="n">sim_awards</span><span class="o">.</span><span class="n">groupby</span><span class="p">(</span><span class="s1">&#39;StartDate&#39;</span><span class="p">)[</span><span class="s1">&#39;AwardedAmountToDate&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">sum</span><span class="p">()</span><span class="o">.</span><span class="n">plot</span><span class="o">.</span><span class="n">bar</span><span class="p">(</span><span class="n">rot</span> <span class="o">=</span> <span class="mi">0</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s1">&#39;Total Awarded Dollars per Year for Awards Similar to Text&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="s1">&#39;Total Dollars Awarded&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">xlabel</span><span class="p">(</span><span class="s1">&#39;Year of Award&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">tight_layout</span><span class="p">()</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre>
  https://github.com/melissajohnsonolson/Chemistry-Award-Matching/blob/master/Figure_2.png
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[15]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">fig3</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">()</span>
<span class="n">sim_papers</span><span class="o">.</span><span class="n">groupby</span><span class="p">(</span><span class="s1">&#39;year&#39;</span><span class="p">)[</span><span class="s1">&#39;title&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">count</span><span class="p">()</span><span class="o">.</span><span class="n">plot</span><span class="o">.</span><span class="n">bar</span><span class="p">(</span><span class="n">rot</span> <span class="o">=</span> <span class="mi">0</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s1">&#39;Number of Publications Each Year </span><span class="se">\n</span><span class="s1"> from Awards Similar to Text&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="s1">&#39;Number of Publications&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">xlabel</span><span class="p">(</span><span class="s1">&#39;Year of Publication&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">tight_layout</span><span class="p">()</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre>https://github.com/melissajohnsonolson/Chemistry-Award-Matching/blob/master/Figure_3.png
  
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[16]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">fig4</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">()</span>
<span class="n">sim_papers</span><span class="o">.</span><span class="n">boxplot</span><span class="p">(</span><span class="n">column</span> <span class="o">=</span> <span class="p">[</span><span class="s1">&#39;citations per year&#39;</span><span class="p">],</span> <span class="n">by</span> <span class="o">=</span>  <span class="s1">&#39;year&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s1">&#39;Citations per Year For </span><span class="se">\n</span><span class="s1"> Publications from Awards Similar to Text&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">suptitle</span><span class="p">(</span><span class="s2">&quot;&quot;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="s1">&#39;Citations per Year&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">xlabel</span><span class="p">(</span><span class="s1">&#39;Year of Publication&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">tight_layout</span><span class="p">()</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre>
https://github.com/melissajohnsonolson/Chemistry-Award-Matching/blob/master/Figure_4.png
