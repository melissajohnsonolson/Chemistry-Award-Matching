# Chemistry-Award-Matching
These sets of scripts allow a user to match their own research description to funded chemistry awards and see the publication trends emerging from those awards



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
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stderr output_text">
<pre>C:\Users\johns\AppData\Local\conda\conda\envs\py36\lib\site-packages\gensim\utils.py:1197: UserWarning: detected Windows; aliasing chunkize to chunkize_serial
  warnings.warn(&#34;detected Windows; aliasing chunkize to chunkize_serial&#34;)
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[2]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">awds</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_csv</span><span class="p">(</span><span class="s1">&#39;NSF CHE 2012.csv&#39;</span><span class="p">,</span> <span class="n">encoding</span><span class="o">=</span><span class="s1">&#39;latin-1&#39;</span><span class="p">)</span>
<span class="n">awds</span><span class="p">[</span><span class="s1">&#39;StartDate&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">to_datetime</span><span class="p">(</span><span class="n">awds</span><span class="p">[</span><span class="s1">&#39;StartDate&#39;</span><span class="p">])</span><span class="o">.</span><span class="n">apply</span><span class="p">(</span><span class="k">lambda</span> <span class="n">x</span><span class="p">:</span> <span class="n">x</span><span class="o">.</span><span class="n">year</span><span class="p">)</span>
<span class="n">awds</span><span class="p">[</span><span class="s1">&#39;EndDate&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">to_datetime</span><span class="p">(</span><span class="n">awds</span><span class="p">[</span><span class="s1">&#39;EndDate&#39;</span><span class="p">])</span>
<span class="n">awds</span><span class="p">[</span><span class="s1">&#39;AwardedAmountToDate&#39;</span><span class="p">]</span><span class="o">=</span><span class="p">[</span><span class="n">x</span><span class="o">.</span><span class="n">replace</span><span class="p">(</span><span class="s1">&#39;$&#39;</span><span class="p">,</span> <span class="s1">&#39;&#39;</span><span class="p">)</span> <span class="k">for</span> <span class="n">x</span> <span class="ow">in</span> <span class="n">awds</span><span class="p">[</span><span class="s1">&#39;AwardedAmountToDate&#39;</span><span class="p">]]</span>
<span class="n">awds</span><span class="p">[</span><span class="s1">&#39;AwardedAmountToDate&#39;</span><span class="p">]</span><span class="o">=</span><span class="p">[</span><span class="n">x</span><span class="o">.</span><span class="n">replace</span><span class="p">(</span><span class="s1">&#39;,&#39;</span><span class="p">,</span> <span class="s1">&#39;&#39;</span><span class="p">)</span> <span class="k">for</span> <span class="n">x</span> <span class="ow">in</span> <span class="n">awds</span><span class="p">[</span><span class="s1">&#39;AwardedAmountToDate&#39;</span><span class="p">]]</span>
<span class="n">awds</span><span class="p">[</span><span class="s1">&#39;AwardedAmountToDate&#39;</span><span class="p">]</span><span class="o">=</span><span class="n">pd</span><span class="o">.</span><span class="n">to_numeric</span><span class="p">(</span><span class="n">awds</span><span class="p">[</span><span class="s1">&#39;AwardedAmountToDate&#39;</span><span class="p">])</span>
<span class="n">awds</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[2]:</div>



<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
<p>5 rows × 25 columns</p>
</div>
</div>

</div>

</div>
</div>

</div>
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

    </div>
</div>
</div>

</div>
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
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>[&#39;project&#39;, &#39;funded&#39;, &#39;chemical&#39;, &#39;synthesis&#39;, &#39;program&#39;, &#39;chemistry&#39;, &#39;division&#39;, &#39;professor&#39;, &#39;tyler&#39;, &#39;mcquade&#39;, &#39;department&#39;, &#39;chemistry&#39;, &#39;biochemistry&#39;, &#39;florida&#39;, &#39;state&#39;, &#39;university&#39;, &#39;develop&#39;, &#39;new&#39;, &#39;synthetic&#39;, &#39;method&#39;, &#39;creating&#39;, &#39;stereoconvergent&#39;, &#39;allylic&#39;, &#39;substitution&#39;, &#39;providing&#39;, &#39;stereoisomer&#39;, &#39;trans&#39;, &#39;substrate&#39;, &#39;catalystcontrolled&#39;, &#39;reactivity&#39;, &#39;neighboring&#39;, &#39;stereocenters&#39;, &#39;limited&#39;, &#39;impact&#39;, &#39;diastereoselectivity&#39;, &#39;regiocontrolled&#39;, &#39;catalytic&#39;, &#39;hydroborations&#39;, &#39;rare&#39;, &#39;metal&#39;, &#39;rhodium&#39;, &#39;iridium&#39;, &#39;become&#39;, &#39;expensive&#39;, &#39;critically&#39;, &#39;important&#39;, &#39;increase&#39;, &#39;range&#39;, &#39;reaction&#39;, &#39;performed&#39;, &#39;inexpensive&#39;, &#39;metal&#39;, &#39;copper&#39;, &#39;resulting&#39;, &#39;method&#39;, &#39;could&#39;, &#39;efficient&#39;, &#39;selective&#39;, &#39;prior&#39;, &#39;reaction&#39;, &#39;could&#39;, &#39;enable&#39;, &#39;new&#39;, &#39;strategy&#39;, &#39;construct&#39;, &#39;valuable&#39;, &#39;complex&#39;, &#39;molecule&#39;, &#39;expensive&#39;, &#39;metal&#39;, &#39;providing&#39;, &#39;basic&#39;, &#39;science&#39;, &#39;necessary&#39;, &#39;improve&#39;, &#39;sustainability&#39;, &#39;proposed&#39;, &#39;work&#39;, &#39;predicted&#39;, &#39;result&#39;, &#39;thorough&#39;, &#39;understanding&#39;, &#39;catalyst&#39;, &#39;provide&#39;, &#39;greater&#39;, &#39;chemical&#39;, &#39;control&#39;, &#39;also&#39;, &#39;creation&#39;, &#39;promising&#39;, &#39;new&#39;, &#39;synthetic&#39;, &#39;method&#39;, &#39;successful&#39;, &#39;result&#39;, &#39;work&#39;, &#39;positively&#39;, &#39;impact&#39;, &#39;pharmaceutical&#39;, &#39;agrochemical&#39;, &#39;specialty&#39;, &#39;chemical&#39;, &#39;industry&#39;, &#39;addition&#39;, &#39;project&#39;, &#39;provide&#39;, &#39;excellent&#39;, &#39;training&#39;, &#39;student&#39;, &#39;undergraduate&#39;, &#39;postdoctoral&#39;, &#39;including&#39;, &#39;group&#39;, &#39;historically&#39;, &#39;underrepresented&#39;, &#39;science&#39;, &#39;student&#39;, &#39;become&#39;, &#39;vanguard&#39;, &#39;leading&#39;, &#39;way&#39;, &#39;future&#39;, &#39;environmentallyfriendly&#39;, &#39;internationallycompetitive&#39;, &#39;organic&#39;, &#39;chemistry&#39;]
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
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
</pre></div>

    </div>
</div>
</div>

</div>
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
    
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>The model correctly matched the training set 92.1% of the time with an average similiarity of 0.875
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[9]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">#Now we can test the model.  Here is a sample abstract from Review Toward Infinitely Recyclable Plastics Derived from</span>
<span class="c1"># Renewable Cyclic Esters in the Journal Chem.</span>

<span class="n">new_text</span> <span class="o">=</span> <span class="s1">&#39;The development of biorenewable and chemically recyclable plastics holds real potential to not only preserve natural resources but also solve the end-of-life issue of plastic waste. However, materializing such potential and ultimately establishing a circular plastics economy requires that three challenges be met: energy cost, depolymerization selectivity, and depolymerizability and performance tradeoffs. Recent advances made in this field, especially the discovery of infinitely recyclable plastics, have yielded feasible solutions and design principles. Future directions will focus on designing monomer and polymer structures that deliver properties and performances for tailored application needs while maintaining complete recyclability and catalyst structures and integrated processes with high (de)polymerization activity, selectivity, and efficiency, ultimately solving the severe worldwide environmental problems created by non-recyclable plastics production and disposal.&#39;</span>
<span class="nb">print</span><span class="p">(</span><span class="n">new_text</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>The development of biorenewable and chemically recyclable plastics holds real potential to not only preserve natural resources but also solve the end-of-life issue of plastic waste. However, materializing such potential and ultimately establishing a circular plastics economy requires that three challenges be met: energy cost, depolymerization selectivity, and depolymerizability and performance tradeoffs. Recent advances made in this field, especially the discovery of infinitely recyclable plastics, have yielded feasible solutions and design principles. Future directions will focus on designing monomer and polymer structures that deliver properties and performances for tailored application needs while maintaining complete recyclability and catalyst structures and integrated processes with high (de)polymerization activity, selectivity, and efficiency, ultimately solving the severe worldwide environmental problems created by non-recyclable plastics production and disposal.
</pre>
</div>
</div>

</div>
</div>

</div>
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
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>The most similar award numbers are 1413033 and 1610311, with similarity scores of 0.625 and 0.613.
The most similar award abstract is:
[&#39;The research group of Dr. Wenjun Du at the Central Michigan University develops new methodologies for preparing a new class of polymers by linking sugar units together and investigates the conditions for controlling the degradation of these polymers.  Sugar-based polymers are potentially compatible with biological systems and useful in biomedical applications.  The degradation of these polymers under controlled conditions to regenerate the monomers allows recycling of the polymer products and, therefore, is environmentally beneficial.  Outreach and educational activities of this project include developing an online course on sustainable polymer for high school teachers, partnering with the Central Michigan Science/Mathematics/Technology Center in organizing outreach activities for K-12 students and teachers, and mentoring graduate and undergraduate students in research.  &lt;br/&gt;&lt;br/&gt;Under the support of Macromolecular, Supramolecular and Nanochemistry Program of NSF, Dr. Wenjun Du aims to develop synthetic methodogies for preparing degradable polymers by connecting sugar units together via orthoester linkages, thereby circumventing the notoriously challenging syntheses of O-glycosyl linkages.  His research group studies the mechanism, scope and limitation of the polymerization process, explores the synthesis of high molecular weight sugar-based polymers by exploiting the reverse anomeric effect, and investigates the degradation of the sugar poly(orthoesters).&#39;]
</pre>
</div>
</div>

</div>
</div>

</div>
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
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[11]:</div>



<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[12]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">#now we pull the relavent award and paper data based on the similarity scores</span>
<span class="n">sim_awards</span> <span class="o">=</span> <span class="n">awds</span><span class="p">[</span><span class="n">awds</span><span class="p">[</span><span class="s1">&#39;AwardNumber&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">isin</span><span class="p">(</span><span class="n">sims</span><span class="p">)]</span><span class="o">.</span><span class="n">copy</span><span class="p">()</span>
<span class="n">sim_papers</span> <span class="o">=</span> <span class="n">papers</span><span class="p">[</span><span class="n">papers</span><span class="p">[</span><span class="s1">&#39;award number&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">isin</span><span class="p">(</span><span class="n">sims</span><span class="p">)]</span><span class="o">.</span><span class="n">copy</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

</div>
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
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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+fjjjwWAFBQUiEj9H2zXSklJEX9/f3FwcJD+/fvLBx98oDXG/PnzJTIyUpydncXV1VViY2O1pmf/2e3TzPv06SP29vbStm3bOpMdSkpK5Pnnnxd/f3+xtbUVf39/efLJJ5V+aydJNDRz788a6vOPP/6QiRMnKtOgAwICZOjQoXL8+HERuTVDbcKECeLu7i6urq4yatQoZcZardpp5ndTXFwsU6dOlc6dO4urq6s4OTlJSEiIvP7661JaWqpsV98kidsfi4j06dNHmcpdq3///loTV+42SeL48ePy8MMPK1P+V69eXWe/AQEBsmjRorv2JiLywgsviLe3d4PTzCMiImTTpk067U+k4Wnm3377rfTp00ecnZ3F2dlZIiMj5ZVXXhERkZ9++kkcHR3l/fffV7ZXq9USGxsrffv2VSYKrV69Wtq0aSPW1tacZm4AKhHeUZfM3/nz59GuXTvs27evzoQLIro/cRYfERGZJQYUERGZJZ7iIyIis8QjKCIiMksMKCIiMktm/T2oixcvmmRcLy8vrdsXWBJL7d1S+wbYO3s3Pn9/f5224xEUERGZJQYUERGZJQYUERGZJQYUERGZJQYUERGZJaPN4rt27RpWr16NX375BSqVCpMnT0ZISIixhiciombGaAGVnp6OLl264NVXX4VarUZVVZWxhiYiombIKKf4rl+/jtzcXOWGeDY2Nk2+yyYREd3fjHIEVVxcDFdXV6xatQo///wzgoKCkJCQUO+twomIiAAjXSz27NmzmDVrFt544w0EBwcjPT0djo6OeOaZZ7S2y8jIQEZGBgAgOTkZ1dXVhi6tXjY2NlCr1SYZ29QstXdL7Rto3r1bHXzOpONruq816fhNYcr33c7OTqftjHIE5enpCU9PTwQHBwMAunfvjq1bt9bZLj4+HvHx8cpjU12Gg5c/sbzeLbVvoHn37mPi8Zvr6wbwUkcKNzc3eHp6KtfWO3HiBFq3bm2MoYmIqJky2iy+8ePHIzU1FWq1Gj4+PpgyZYqxhiYiombIaAEVGBiI5ORkYw1HRETNHK8kQUREZokBRUREZokBRUREZokBRUREZokBRUREZokBRUREZokBRUREZokBRUREZokBRUREZokBRUREZokBRUREZokBRUREZokBRUREZokBRUREZokBRUREZokBRUREZokBRUREZokBRUREZokBRUREZokBRUREZokBRUREZokBRUREZokBRUREZokBRUREZokBRUREZokBRUREZsnGWAMlJibCwcEBVlZWsLa2RnJysrGGJiKiZshoAQUAc+fOhaurqzGHJCKiZoqn+IiIyCypRESMMVBiYiJcXFwAAAMGDEB8fHydbTIyMpCRkQEASE5ORnV1tTFKq8PGxgZqtdokY5uapfZuqX0Dzbt3q4PPmXR8Tfe1Jh2/KUz5vtvZ2em0ndECqrS0FB4eHigrK8OCBQswbtw4RERENPicixcvGqO0Ory8vFBSUmKSsU3NUnu31L6B5t27z5kZJh2/uMMik47fFKZ83/39/XXazmin+Dw8PAAALVu2RHR0NM6cOWOsoYmIqBkySkBVVlbixo0bys8//vgj2rRpY4yhiYiomTLKLL6ysjIsXboUAFBTU4NevXqhS5cuxhiaiIiaKaMElK+vL5YsWWKMoYiI6D7BaeZERGSWGFBERGSWGFBERGSWdAqoq1evorKyEgCg0Wiwd+9eZGZmQqPRGLQ4IiKyXDoFVHJyMgoLCwEAmzZtwvbt27Fjxw5s2LDBoMUREZHl0imgCgsLERgYCADYt28fZs6ciblz5+K7774zZG1ERGTBdJpmbmVlBbVajcLCQjg5OcHLywsajUY57UdERKRvOgVUly5dsHz5cpSXl6Nnz54AgF9//VW5fBEREZG+6RRQzz//PDIzM2FtbY3Y2FgAQHl5OYYNG2bQ4oiIyHLpFFC2trZ1bo8RGRlpkIKIiIiABgJqxYoVUKlUd93BCy+8oNeCiIiIgAZm8fn5+cHX1xe+vr5wcnLCoUOHoNFo4OHhAY1Gg0OHDsHJycmYtRIRkQW54xHU7Z8vLVy4ENOnT0d4eLiyLC8vD//7v/9r2OqIiMhi6fQ9qFOnTiE4OFhrWYcOHXDq1CmDFEVERKRTQLVr1w6bNm1CdXU1AKC6uhqbN29WvrxLRESkbzrN4psyZQpSU1MxduxYuLi4oKKiAu3bt8dLL71k6PqIiMhC3TWgRAQignnz5uHy5cu4fPky3N3d4eXlZYz6iIjIQt31FJ9KpcJ///d/Q6VSwcvLC8HBwQwnIiIyOJ0+gwoMDFSuZk5ERGQMOn0GFRkZiTfffBN9+vSpc/QUFxdnkMKIiMiy6RRQ+fn58PHxQW5ubp11DCgiIjIEnQJq7ty5hq6DiIhIi04BdbvaWX21rKx0+hiLiIioUXQKqNLSUqxbtw65ubm4du2a1rpPPvnEIIUREZFl0+nw591334WNjQ2SkpLg4OCAxYsXIyoqCs8995yh6yMiIgul87X4Jk+ejMDAQKhUKgQGBmLy5MnYsWNHowbTaDT4xz/+geTk5HsqloiILIdOAWVlZQVra2sAgLOzM65evQp7e3uUlpY2arCvvvoKAQEBja+SiIgsjk4B1aFDBxw7dgwA8OCDD2L58uVYunQp2rdvr/NAly5dwtGjR9G/f/97q5SIiCyKTpMkXnzxRWXmXkJCArZv344bN25g8ODBOg/0wQcfYPTo0bhx48Ydt8nIyEBGRgYAIDk52WSXVLKxsWm2l3OKeXu/ycbOntrLZGMDQHramSY8+0qTxh6X2KFJzzel5vz7jqa85XrQbF83NI/3XaeAcnZ2Vn62s7PD3/72t0YNcuTIEbRs2RJBQUE4efLkHbeLj49HfHy88rikpKRR4+iLl5eXycZuziz5NWvOvTfn33cfE4/fXF83wLTvu7+/v07b6RRQ48ePR3h4OCIiIhAREaFMltBVfn4+Dh8+jGPHjqG6uho3btxAamoqb9dBRER3pFNAvfnmm8jNzUVOTg6++uorXL9+HaGhoYiIiMCQIUPu+vyRI0di5MiRAICTJ09i+/btDCciImqQTgHl5+cHPz8/9OvXDxcvXkRWVhZ27tyJH374QaeAIiIiaiydAuqbb75BTk4O8vPz4eHhgfDwcEydOhVhYWGNHjAyMhKRkZGNfh4REVkWnQJq3bp18PX1xd/+9jd07doV7u7uhq6LiIgsnE4B9c477yAnJwe5ubnYuXMn1Gq1MmkiNjbW0DUSEZEF0imgPDw80KtXL/Tq1QsFBQU4ePAgvv76a+zdu5cBRUREBqFTQO3YsUM5gnJwcEBERATGjBmDiIgIQ9dHREQWSqeA+vnnnxEdHY2xY8fC19fX0DURERHpFlCJiYl1llVUVCA7OxsDBw7Ue1FERESNuqOuRqPB0aNHkZmZiaNHj8LPz48BRUREBqFTQJ07dw5ZWVnIzs5GdXU1bt68iVdeeQVRUVGGro+IiCxUgwG1bds2ZGZmoqioCJ07d0ZCQgKioqLw4osvIjg42Fg1EhGRBWowoD766CO4uLggMTERPXr0aNQFYomIiJqiwYBKSkpCZmYm1qxZg/Xr1yMmJga9evViUBERkcE1GFC118179tlncfDgQWRmZuKrr76CiODbb7/FwIED0aJFC2PVSkREFkSnSRJ2dnaIjY1FbGwsLl26hMzMTOzbtw9bt27Fxo0bDV0jERFZoEZNMwcAT09PDB06FEOHDsXp06cNURMRERGsmvJkzuQjIiJDaVJAERERGQoDioiIzNIdA2rWrFnKz5999plRiiEiIqp1x4C6ePEiqqurAdy63QYREZEx3XEWX3R0NKZOnQofHx9UV1dj7ty59W43b948gxVHRESW644BNWXKFOTl5aG4uBhnzpxBv379jFkXERFZuAa/BxUWFoawsDCo1Wr07dvXSCURERHp+EXduLg4/PTTT8jKysLly5fh7u6O2NhYdOzY0dD1ERGRhdJpmvnu3buRkpICNzc3dOvWDe7u7nj77beRkZFh6PqIiMhC6XQEtW3bNsyePRuBgYHKsp49e2LZsmWIj483VG1ERGTBdAqo8vJytG7dWmuZv78/KioqdBqkdhagWq1GTU0NunfvjuHDhze+WiIishg6BVRYWBg2bNiAUaNGwd7eHpWVlfj4448REhKi0yC2traYO3cuHBwcoFarkZSUhC5duuj8fCIisjw6BdRzzz2HlJQUJCQkwMXFBRUVFQgJCcHUqVN1GkSlUsHBwQEAUFNTg5qaGt70kIiIGqRTQLm7u2PevHm4dOmSMovP09OzUQNpNBq8/vrrKCoqwsCBA+u9EnpGRoYy8SI5ORleXl6NGkNfbGxsTDZ2c2b61+yKyUY2de9JSUkmG3v+/PkmGxtnTDc0YNr3PS1zkMnGTuyz0yjjNOp+UJ6eno0OplpWVlZYsmQJrl27hqVLl+LChQto06aN1jbx8fFaky5KSkruaaym8vLyMtnYzZklv2bs3TR8TDbyLZb6vje1b39/f522M/rVzJ2dnREREYHjx48be2giImpGjBJQV69exbVr1wDcmtF34sQJBAQEGGNoIiJqpu56ik+j0SAnJwdhYWGwsWn0HeIBAJcvX0ZaWho0Gg1EBD169EDXrl3vaV9ERGQZ7po4VlZWeOutt7Bhw4Z7HqRt27Z466237vn5RERkeXQ6xRceHo5Tp04ZuhYiIiKFTufsvL29sWjRIkRFRcHT01PrO0xPP/20wYojIiLLpVNAVVdXIzo6GgBQWlpq0IKIiIgAHQNqypQphq6DiIhIi87T8n799VccPHgQZWVlmDBhAi5evIibN2+ibdu2hqyPiIgslE6TJA4cOIC5c+eitLQUWVlZAIAbN240aWYfERFRQ3Q6gvr0008xZ84cBAYG4sCBAwBuTR0/f/68IWsjIiILptMRVFlZWZ1TeSqVilckJyIig9EpoIKCgpRTe7Wys7PRoUMHgxRFRESk0ym+cePGYcGCBdizZw+qqqqwcOFCXLx4EbNnzzZ0fUREZKF0CqiAgACkpKTgyJEj6Nq1Kzw9PdG1a1flJoRERET6pvM0c3t7e4SFhaG0tBQeHh4MJyIiMiidAqqkpASpqak4ffo0nJ2dce3aNXTo0AEvvfQSvL29DV0jERFZIJ0mSaSlpSEoKAjp6el47733kJ6ejvbt2yMtLc3Q9RERkYXSKaDOnTuH0aNHK6f1HBwcMHr0aJw7d86gxRERkeXSKaCCg4Nx5swZrWVnz55FSEiIQYoiIiK642dQn3zyifKzr68vFi1ahIceegienp64dOkSjh07hl69ehmlSCIisjx3DKhLly5pPX744YcBAFevXoWtrS26deuG6upqw1ZHREQW644BxVtsEBGRKen8PaiqqioUFRWhsrJSa3loaKjeiyIiItIpoDIzM/H+++/DxsYGdnZ2WuveeecdgxRGRESWTaeA2rhxI1599VV07tzZ0PUQEREB0HGauY2NDSIiIgxdCxERkUKngHr66aexYcMGXL161dD1EBERAdDxFJ+/vz8+/fRT7Nq1q866278vRUREpC86BdSKFSsQGxuLnj171pkkoYuSkhKkpaXhypUrUKlUiI+Px2OPPdbo/RARkeXQKaAqKirw9NNP3/Mt3q2trTFmzBgEBQXhxo0bmD59Ojp37ozWrVvf0/6IiOj+p9NnUH379q1zy/fGcHd3R1BQEADA0dERAQEBKC0tvef9ERHR/U+nI6gzZ87g66+/xhdffAE3NzetdfPmzWvUgMXFxSgoKECHDh3qrMvIyEBGRgYAIDk5GV5eXo3at77Y2NiYbOzmzPSv2RWTjWz63k3HpL2fufsmhmSp77ux+tYpoPr374/+/fs3ebDKykosW7YMCQkJcHJyqrM+Pj4e8fHxyuOSkpImj3kvvLy8TDZ2c2bJrxl7Nw0fk418i6W+703t29/fX6ftdAqovn37NqUWAIBarcayZcvQu3dv5cKzREREd6JTQO3Zs+eO6+Li4u76fBHB6tWrERAQgMcff1z36oiIyGLpFFD79u3TenzlyhUUFRUhLCxMp4DKz89HVlYW2rRpg9deew0AMGLECDz00EP3UDIREVkCnQJq7ty5dZbt2bMHv/32m06DhIWF4dNPP21cZUREZNF0mmZen759+zZ46o+IiKgpdDqC0mg0Wo+rq6uRlZUFZ2dngxRFRESkU0CNGDGizjIPDw9MmjRJ7wUREREBOgbUypUrtR7b29vD1dXVIAUREREBOgaUt7e3oesgIiLS0mBA3e0yRiqVCklJSXotiIiICLhLQPXu3bvrd1UqAAAPAUlEQVTe5aWlpdi5cyeqqqoMUhQREVGDAfXnL+GWl5djy5Yt2L17N3r27ImnnnrKoMUREZHl0ukzqOvXr2Pbtm3YtWsXHnroISxevBh+fn6Gro2IiCxYgwFVXV2Nf/3rX9ixYwciIiIwf/58PPDAA8aqjYiILFiDAZWYmAiNRoMhQ4agffv2KCsrQ1lZmdY2HTt2NGiBRERkmRoMKDs7OwDAN998U+96lUpV5ztSRERE+tBgQKWlpRmrDiIiIi33fLFYIiIiQ2JAERGRWWJAERGRWWJAERGRWWJAERGRWWJAERGRWWJAERGRWWJAERGRWWJAERGRWWJAERGRWWJAERGRWWJAERGRWdLphoVNtWrVKhw9ehQtW7bEsmXLjDEkERE1c0Y5gurbty9mzpxpjKGIiOg+YZSAioiIgIuLizGGIiKi+wQ/gyIiIrNklM+gdJWRkYGMjAwAQHJyMry8vO5pP7//tWeT6vi9Sc8GfLd818Q9NE/3+n7pzxWTjWz63k3HpL2fMd3QgOW+78bq26wCKj4+HvHx8crjkpISE1Zz75pr3U1lqX0D7N1UfEw28i2W+r43tW9/f3+dtuMpPiIiMktGOYJKSUlBTk4OysvL8fzzz2P48OGIi4szxtBERNRMGSWgpk2bZoxhiIjoPsJTfEREZJYYUEREZJYYUEREZJYYUEREZJYYUEREZJYYUEREZJYYUEREZJYYUEREZJYYUEREZJYYUEREZJYYUEREZJYYUEREZJYYUEREZJYYUEREZJYYUEREZJYYUEREZJYYUEREZJYYUEREZJYYUEREZJYYUEREZJYYUEREZJYYUEREZJYYUEREZJYYUEREZJYYUEREZJYYUEREZJZsjDXQ8ePHkZ6eDo1Gg/79++PJJ5801tBERNQMGeUISqPRYN26dZg5cyaWL1+O7Oxs/Prrr8YYmoiImimjBNSZM2fg5+cHX19f2NjYoGfPnjh06JAxhiYiomZKJSJi6EEOHjyI48eP4/nnnwcAZGVl4fTp05gwYYLWdhkZGcjIyAAAJCcnG7osIiIyY0Y5gqovA1UqVZ1l8fHxSE5ONnk4TZ8+3aTjm5Kl9m6pfQPs3VI1h96NElCenp64dOmS8vjSpUtwd3c3xtBERNRMGSWg2rdvj8LCQhQXF0OtVuO7775DVFSUMYYmIqJmyvqf//znPw09iJWVFfz8/LBixQp8/fXX6N27N7p3727oYZskKCjI1CWYjKX2bql9A+zdUpl770aZJEFERNRYvJIEERGZJQYUERGZJaNd6siUSkpKkJaWhitXrkClUiE+Ph6PPfYYKioqsHz5cvzxxx/w9vbGyy+/DBcXF/z2229YtWoVCgoK8Mwzz2DIkCEN7sec6av36upqzJ07F2q1GjU1NejevTuGDx9u4u4apq/ea2k0GkyfPh0eHh5mPUVXn30nJibCwcEBVlZWsLa2NvlXQO5Gn71fu3YNq1evxi+//AKVSoXJkycjJCTEhN01TF+9X7x4EcuXL1f2W1xcjOHDh2Pw4MHGb0osQGlpqZw9e1ZERK5fvy4vvfSS/PLLL/Lhhx/Kli1bRERky5Yt8uGHH4qIyJUrV+T06dPy8ccfy5dffnnX/ZgzffWu0Wjkxo0bIiJy8+ZNmTFjhuTn5xu5m8bRV++1tm/fLikpKbJo0SLjNXEP9Nn3lClTpKyszLgNNIE+e1+xYoVkZGSIyK3f+YqKCiN20nj6/n0XEampqZFnn31WiouLjdPEn1jEKT53d3dltoqjoyMCAgJQWlqKQ4cOoU+fPgCAPn36KJdfatmyJTp06ABra2ud9mPO9NW7SqWCg4MDAKCmpgY1NTX1ftnanOird+DWd/eOHj2K/v37G6+Be6TPvpsbffV+/fp15ObmIi4uDgBgY2MDZ2dnI3bSeIZ430+cOAE/Pz94e3sbvoF6WMQpvtsVFxejoKAAHTp0QFlZmfKFYXd3d1y9evWe9tNcNLV3jUaD119/HUVFRRg4cCCCg4MNXbLeNLX3Dz74AKNHj8aNGzcMXape6eP3feHChQCAAQMGID4+3mC16ltTei8uLoarqytWrVqFn3/+GUFBQUhISFD+k2bu9PXvXHZ2NmJiYgxV5l1ZxBFUrcrKSixbtgwJCQlwcnIy+X6MSR81W1lZYcmSJVi9ejXOnj2LCxcu6LlKw2hq70eOHEHLli3N/jsjf6aP9/yNN97A4sWLMXPmTOzatQs5OTl6rtIwmtp7TU0NCgoK8Mgjj+Ctt96Cvb09tm7daoBK9U9f/z6p1WocOXLEpN9ZtZiAUqvVWLZsGXr37o2HH34YwK1D3MuXLwMALl++DFdX13vaj7nTV++1nJ2dERERgePHjxukXn3SR+/5+fk4fPgwEhMTkZKSgp9++gmpqakGr70p9PWee3h4KM+Njo7GmTNnDFe0nuijd09PT3h6eipnCbp3746CggLDFq4H+vy7fuzYMbRr1w5ubm4Gq/duLCKgRASrV69GQEAAHn/8cWV5VFQUMjMzAQCZmZmIjo6+p/2YM331fvXqVVy7dg3ArRl9J06cQEBAgOEK1wN99T5y5EisXr0aaWlpmDZtGjp27IiXXnrJoLU3hb76rqysVE5pVlZW4scff0SbNm0MV7ge6Kt3Nzc3eHp64uLFiwBufRbTunVrwxWuB/rqvZapT+8BFnIliby8PCQlJaFNmzbKB/sjRoxAcHAwli9fjpKSEnh5eeGVV16Bi4sLrly5gunTp+PGjRvK5ID/+Z//wYULF+rdz0MPPWTK9hqkr97/+OMPpKWlQaPRQETQo0cPPPXUUyburmH66v320yQnT57E9u3bzXqaub76Li8vx9KlSwHcOuXVq1cvDB061JSt3ZU+3/Pz589j9erVUKvV8PHxwZQpU+Di4mLiDu9Mn71XVVVh8uTJWLlypUk/xrCIgCIioubHIk7xERFR88OAIiIis8SAIiIis8SAIiIis8SAIiIis8SAIjIBEcHKlSuRkJCA2bNnm7qcBm3evBlpaWmmLoMsEAOK7lupqalYtWqV1rKcnByMHz9e+Wa9qZw8eRI5OTlYs2YNFixYcMftfvzxRwwfPhw7duwwYnVE5oEBRfetcePG4dixY/jxxx8B3LoCxpo1a/D3v/9duXimvmg0mkZtX1JSAh8fH9jb2ze4XWZmJlxcXJQrARiSRqNpdB9EhmRxVzMny9GiRQuMHz8ea9aswbJly/DFF1/A19cXffv2BXDrH+StW7di7969uH79Ojp16oRnn30WLi4u0Gg0WL58OfLy8nDz5k0EBgbi2WefVS53k5qaCicnJ/z+++/Iy8vD9OnTERkZqTX+pUuXsHbtWuTn56NFixZ48sknERcXh4yMDKSnp0OtVmPMmDF44okn6r0qR2VlJf7zn/9g8uTJWLFiBc6fP4/AwEBl/A4dOuCxxx7DH3/8gcTEREycOBHx8fH47bffkJSUhPfeew8VFRVYuXIlzpw5A41Gg9DQUEycOFG5xt6cOXMQGRmJEydO4Pz581i+fDlEBGlpaTh//jxCQ0Ph4+NjuDeJqAE8gqL7Wo8ePRAUFIS3334bGRkZmDhxorJux44dOHbsGObNm4d33nkH9vb2SE9PV9Z37doVqampePfdd/HAAw9g5cqVWvvOzs7GsGHDsH79+nrvtJqSkgIfHx+sWbMG06ZNw0cffYScnBzEx8dj/PjxCA8Px4cffnjHS0YdOHAAzs7O6N69Ozp16qR1FBUREYGTJ08CuHXa0tfXV7nSeG5uLsLDw6FSqSAi6N+/P9555x2kpaXBxsZGq0cA2LdvHyZPnoz169fD09MTKSkpCA4Oxrp16/DEE08gKyurka86kX4woOi+N2HCBPz000946qmn4OXlpSzPyMjAiBEj4OHhATs7OwwbNgwHDhyARqOBlZUV+vbtC0dHR2XduXPnUFlZqTw/OjoaISEhsLKygq2trdaYxcXFOHPmDEaOHAk7OzsEBQWhb9++jfrHPjMzEz179oSVlRV69eqF/fv3o6amBsCtgMrNzYWIIDc3F0888QRyc3MB3AqsiIgIAICrqyu6desGOzs7ODk54a9//WudW2b069cPrVu3ho2NDUpKSnD+/HkMHz4ctra26NixI/7rv/6rcS84kZ7wFB/d99zc3ODq6lrnatQlJSVYvHix1p2BVSoVrl69CldXV3z88cc4ePAgysvLlW3Ky8uVm9bdHnZ/VlpaihYtWmjd4M7b2xu//PKLTjUXFxcjNzcXY8eOBQB069YNa9euxfHjx9G1a1f4+/vDxsYGFy5cQG5uLp555hl8++23KCoqQk5ODoYMGQLg1mnCDz74AD/88AOuX78OAHVuuujp6an8fPnyZbRo0ULrszFvb+9G3eSOSF8YUGSxPD098dJLL9V7Z+C9e/fi2LFjSEpKgre3N8rLy/Hss89C12sre3h4oLy8HJWVlUpIlZSUKJ/93E1WVhZEBG+++aayTK1WIzMzE127dgUAhIeHIzs7GyqVCm5ubggPD8eePXtQVVWl3BZj27ZtKC4uxqJFi+Dm5oazZ89ixowZWmPdHtBubm4oLy9HdXU17OzslLprfyYyJp7iI4s1YMAAbNq0CSUlJQCAsrIyHD58GMCtowwbGxu0aNECVVVV2Lx5c6P27ePjg6CgIGzatAk3b97E+fPnsXfvXvTq1Uun52dmZmL48OFYsmSJ8mfatGk4cuQIKioqANw6zbdr1y7ldF5kZCR27dqF8PBwWFlZKX3Y2dnB2dkZ5eXl+Pzzzxsc18/PD23btsWnn34KtVqNnJwcHD16tFG9E+kLj6DIYtXe1G3+/Pm4cuUKWrZsiZiYGERFRaFfv3748ccfMWnSJLRo0QLDhg1DRkZGo/b/8ssvY+3atZg4cSJcXFwwYsQIdOzY8a7Py8vLw+XLl/Hoo49q3X+oW7du8Pb2xnfffYdHHnkEERERuHHjBsLDwwHcOqKqrKxUHtf2mJqaivHjx8PDwwODBw/GkSNHGhx/2rRpSEtLw7hx4xAWFobY2FhUV1c3qncifeD9oIiIyCzxFB8REZklBhQREZklBhQREZklBhQREZklBhQREZklBhQREZklBhQREZklBhQREZml/wPccBhf0+49cQAAAABJRU5ErkJggg==
"
>
</div>

</div>

</div>
</div>

</div>
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
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>
</div>

</div>
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
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>
</div>

</div>
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
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>




<div class="output_text output_subarea ">
<pre>&lt;Figure size 432x288 with 0 Axes&gt;</pre>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span> 
</pre></div>

    </div>
</div>
</div>

</div>
    </div>
  </div>
</body>

 


</html>
