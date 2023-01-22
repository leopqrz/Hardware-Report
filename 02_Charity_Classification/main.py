"""docstring."""
import pickle

import nltk
import numpy as np
import pandas as pd
import streamlit as st
import utils
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')

feat = ['100', '12', '18', '1971', '1972', '1976', '1977', '1981', '1982',
        '1983', '1984', '1985', '1986', '1987', '1988', '1989', '1990',
        '1991', '1992', '1993', '1995', '20', '200', '2000', '25', '30',
        '40', '50', '501c3', '60', 'ability', 'able', 'abuse', 'abused',
        'academic', 'academy', 'access', 'accessible', 'accomplish',
        'achieve', 'achievement', 'acres', 'across', 'across country',
        'act', 'action', 'active', 'actively', 'activities', 'addition',
        'address', 'addressing', 'adoption', 'adult', 'adults', 'advance',
        'advances', 'advancing', 'advocacy', 'advocate', 'advocates',
        'advocating', 'affected', 'affiliate', 'affordable',
        'affordable housing', 'afterschool', 'age', 'agencies', 'agency',
        'ages', 'aid', 'air', 'alliance', 'almost', 'along', 'also', 'america',
        'american', 'americans', 'americas', 'among', 'angeles', 'animal',
        'animals', 'annual', 'annually', 'appreciation', 'approach',
        'approximately', 'area', 'areas', 'around', 'around world', 'array',
        'art', 'artistic', 'artists', 'arts', 'assist', 'assistance',
        'assists', 'association', 'atrisk', 'audience', 'audiences',
        'available', 'awardwinning', 'awareness', 'back', 'backgrounds',
        'bank', 'barriers', 'based', 'basic', 'basic needs', 'bay', 'become',
        'began', 'belief', 'believe', 'believes', 'benefit', 'benefits',
        'best', 'better', 'beyond', 'bible', 'biblical', 'big', 'blocks',
        'board', 'body', 'books', 'boy', 'boy scouts', 'boys', 'boys girls',
        'break', 'break cycle', 'bring', 'bringing', 'brings', 'broad',
        'build', 'building', 'builds', 'built', 'business', 'businesses',
        'california', 'call', 'camp', 'campaign', 'campaigns', 'camps',
        'campus', 'cancer', 'capacity', 'capital', 'care', 'career',
        'careers', 'caring', 'carolina', 'case', 'case management', 'catalyst',
        'catholic', 'causes', 'center', 'centers', 'central', 'century',
        'challenge', 'challenges', 'change', 'change lives', 'changes',
        'changing', 'character', 'charge', 'charitable', 'charities',
        'charity', 'chicago', 'child', 'childhood', 'children',
        'children adults', 'children families', 'children youth',
        'childrens', 'choices', 'christ', 'christian', 'church', 'churches',
        'cities', 'citizens', 'city', 'citys', 'civic', 'civil', 'classes',
        'classical', 'classroom', 'clean', 'clients', 'clinic', 'clinical',
        'clothing', 'club', 'clubs', 'coalition', 'collaboration',
        'collaborative', 'collection', 'collections', 'college', 'come',
        'commitment', 'committed', 'common', 'common good', 'communities',
        'community', 'community leaders', 'community outreach',
        'communitybased', 'communitys', 'companies', 'company', 'compassion',
        'compassionate', 'complex', 'comprehensive', 'concerned', 'concerts',
        'conditions', 'confidence', 'connect', 'connecting', 'connections',
        'connects', 'conservation', 'contemporary', 'continue', 'continues',
        'continuing', 'contribute', 'contributions', 'core', 'cost', 'council',
        'counseling', 'counties', 'countries', 'country', 'county', 'countys',
        'create', 'created', 'creates', 'creating', 'creation', 'creative',
        'crisis', 'critical', 'cruelty', 'cultural', 'culture', 'cultures',
        'cure', 'current', 'currently', 'curriculum', 'cycle', 'daily',
        'dance', 'day', 'days', 'dc', 'decisions', 'dedicated', 'deliver',
        'department', 'design', 'designed', 'develop', 'developed',
        'developing', 'development', 'develops', 'devoted', 'difference',
        'different', 'dignity', 'direct', 'directly', 'disabilities',
        'disadvantaged', 'discover', 'discovery', 'disease', 'distribute',
        'distributed', 'distribution', 'district', 'diverse', 'diversity',
        'dogs', 'domestic', 'domestic violence', 'donated', 'donations',
        'donors', 'doors', 'downtown', 'dynamic', 'early', 'east',
        'economic', 'economy', 'educate', 'educates', 'educating', 'education',
        'education advocacy', 'education community', 'education income',
        'education programs', 'educational', 'educational programs',
        'educators', 'effective', 'effectively', 'effort', 'efforts',
        'elderly', 'emergency', 'emerging', 'emotional', 'employment',
        'empower', 'empowering', 'empowers', 'enable', 'encourage',
        'encouraging', 'end', 'energy', 'engage', 'engaged', 'engagement',
        'engages', 'engaging', 'enhance', 'enjoy', 'enrich', 'enrichment',
        'ensure', 'ensuring', 'entire', 'environment', 'environmental',
        'equip', 'especially', 'essential', 'established', 'ethical', 'events',
        'every', 'every child', 'every day', 'everyone', 'excellence',
        'exceptional', 'exhibitions', 'exhibits', 'exists', 'expand',
        'expanded', 'experience', 'experiences', 'expertise', 'explore',
        'extensive', 'facilities', 'facility', 'facing', 'faith', 'faithbased',
        'families', 'families children', 'families communities',
        'families individuals', 'family', 'federal', 'federation', 'feeding',
        'field', 'fight', 'financial', 'financial assistance', 'find', 'first',
        'fitness', 'five', 'florida', 'focus', 'focused', 'focuses',
        'focusing', 'following', 'food', 'food bank', 'food clothing',
        'force', 'form', 'formed', 'formerly', 'foster', 'fostering',
        'foundation', 'foundations', 'founded', 'founding', 'four', 'free',
        'free charge', 'freedom', 'friends', 'fulfill', 'full',
        'full potential', 'fun', 'fund', 'funding', 'fundraising',
        'funds', 'future', 'future generations', 'gardens', 'general',
        'generation', 'generations', 'georgia', 'get', 'gifts', 'girls',
        'girls clubs', 'give', 'given', 'giving', 'global', 'go', 'goal',
        'goals', 'god', 'gods', 'good', 'gospel', 'government', 'graduate',
        'grants', 'grassroots', 'great', 'greater', 'greatest', 'group',
        'groups', 'grow', 'growing', 'grown', 'growth', 'guidance', 'guided',
        'habitat', 'habitat humanity', 'hall', 'handson', 'healing',
        'health', 'health care', 'healthcare', 'healthy', 'heart', 'help',
        'help people', 'helped', 'helping', 'helps', 'heritage', 'high',
        'high quality', 'high school', 'higher', 'highest', 'highquality',
        'historic', 'historical', 'history', 'home', 'homeless',
        'homelessness', 'homes', 'hope', 'hospital', 'hours', 'house',
        'houses', 'housing', 'human', 'human service', 'human services',
        'humane', 'humane society', 'humanity', 'hundreds', 'hunger',
        'hungry', 'ideas', 'identify', 'illness', 'immediate', 'impact',
        'importance', 'important', 'improve', 'improve lives',
        'improve quality', 'improving', 'inc', 'inception', 'include',
        'includes', 'including', 'inclusive', 'income', 'incorporated',
        'increase', 'increasing', 'independence', 'independent', 'individual',
        'individuals', 'individuals families', 'industry', 'influence',
        'information', 'informed', 'initiative', 'initiatives', 'innovation',
        'innovative', 'inspire', 'inspired', 'inspires', 'inspiring',
        'institute', 'institution', 'institutions', 'integrity',
        'intellectual', 'interactive', 'interest', 'interests',
        'international', 'internationally', 'intervention', 'involved',
        'involvement', 'island', 'israel', 'issues', 'items', 'jersey',
        'jesus', 'jesus christ', 'jewish', 'jewish community', 'job',
        'joy', 'justice', 'keep', 'key', 'kids', 'knowledge', 'known',
        'land', 'large', 'largest', 'last', 'lasting', 'law', 'lead',
        'leader', 'leaders', 'leadership', 'leadership development',
        'leading', 'leads', 'learn', 'learning', 'led', 'legacy',
        'legal', 'level', 'levels', 'library', 'life', 'life skills',
        'lifechanging', 'lifelong', 'like', 'literacy', 'live', 'lives',
        'living', 'local', 'locally', 'located', 'locations', 'long',
        'longterm', 'los', 'los angeles', 'lost', 'love', 'loving',
        'low', 'lowincome', 'made', 'main', 'maintain', 'major',
        'make', 'makes', 'making', 'management', 'many', 'materials',
        'may', 'meals', 'meaningful', 'means', 'media', 'medical',
        'medical care', 'meet', 'meeting', 'member', 'members',
        'membership', 'men', 'men women', 'mental', 'mental health',
        'mentoring', 'metropolitan', 'michigan', 'middle', 'military',
        'million', 'millions', 'mind', 'ministries', 'ministry',
        'mission', 'mission provide', 'mobilize', 'mobilizing', 'model',
        'money', 'move', 'movement', 'much', 'museum', 'museums', 'music',
        'musical', 'name', 'nation', 'national', 'national international',
        'nationally', 'nations', 'nationwide', 'native', 'natural', 'nature',
        'nearly', 'necessary', 'need', 'need us', 'needed', 'needs', 'needy',
        'neglect', 'neglected', 'neighborhood', 'neighborhoods', 'neighbors',
        'network', 'new', 'new jersey', 'new york', 'news', 'next',
        'nonpartisan', 'nonprofit', 'nonprofit organization',
        'nonprofit organizations', 'nonprofits', 'north', 'northern',
        'number', 'nurture', 'nurturing', 'nutrition', 'nutritious',
        'offer', 'offered', 'offering', 'offers', 'often', 'oldest', 'one',
        'ongoing', 'open', 'opened', 'opera', 'operate', 'operates',
        'operating', 'opportunities', 'opportunity', 'orchestra', 'order',
        'organization', 'organization dedicated', 'organizations', 'organized',
        'original', 'others', 'otherwise', 'outdoor', 'outreach',
        'outreach programs', 'outstanding', 'pantries', 'parent',
        'parents', 'park', 'part', 'participate', 'participation',
        'partner', 'partners', 'partnership', 'partnerships', 'passion',
        'past', 'patient', 'patients', 'people', 'people ages', 'people need',
        'peoples', 'per', 'performance', 'performances', 'performing',
        'performing arts', 'permanent', 'person', 'personal', 'persons',
        'pet', 'pets', 'philanthropic', 'philanthropy', 'physical',
        'place', 'places', 'planning', 'play', 'policies', 'policy',
        'political', 'poor', 'population', 'positive', 'possible', 'potential',
        'potential productive', 'pounds', 'poverty', 'power', 'powerful',
        'practical', 'practice', 'practices', 'premier', 'prepare', 'present',
        'presenting', 'presents', 'preservation', 'preserve', 'preserving',
        'prevent', 'prevention', 'primarily', 'primary', 'principles',
        'private', 'problems', 'process', 'produce', 'productions',
        'productive', 'professional', 'professionals', 'program',
        'programming', 'programs', 'programs include', 'programs services',
        'project', 'projects', 'promote', 'promotes', 'promoting', 'protect',
        'protecting', 'protection', 'provide', 'provided', 'provider',
        'provides', 'providing', 'public', 'public education', 'public policy',
        'publications', 'purpose', 'pursue', 'put', 'quality', 'quality life',
        'radio', 'raise', 'raised', 'raising', 'range', 'reach', 'reach full',
        'reaches', 'reaching', 'real', 'realize', 'receive', 'recognized',
        'recovery', 'recreation', 'recreational', 'reduce', 'regardless',
        'region', 'regional', 'regions', 'rehabilitation', 'related',
        'relationship', 'relationships', 'relief', 'religious', 'rescue',
        'research', 'research education', 'residential', 'residents',
        'resource', 'resources', 'respect', 'response', 'responsibility',
        'responsible', 'restoration', 'restore', 'results', 'rich',
        'right', 'rights', 'risk', 'river', 'role', 'safe', 'safety',
        'san', 'save', 'scholarship', 'scholarships', 'school', 'schools',
        'science', 'scientific', 'scientists', 'scouts', 'scouts america',
        'second', 'secure', 'security', 'see', 'seek', 'seeking', 'seeks',
        'selfsufficiency', 'senior', 'seniors', 'sense', 'series', 'serve',
        'served', 'serves', 'service', 'services', 'serving', 'set', 'setting',
        'seven', 'sexual', 'share', 'shared', 'sharing', 'shelter', 'shelters',
        'significant', 'simple', 'since', 'since founding', 'since inception',
        'single', 'sites', 'six', 'skills', 'small', 'social',
        'social service', 'social services', 'society', 'solutions',
        'sound', 'source', 'south', 'southern', 'space', 'special',
        'spirit', 'spiritual', 'sports', 'st', 'stability', 'stable',
        'staff', 'standards', 'state', 'states', 'statewide', 'stations',
        'stay', 'stewardship', 'strategic', 'strategies', 'street',
        'strength', 'strengthen', 'strengthening', 'strive',
        'strives', 'strong', 'stronger', 'student', 'students',
        'studies', 'study', 'succeed', 'success', 'successful', 'suffering',
        'summer', 'support', 'support services', 'supported', 'supporting',
        'supportive', 'supports', 'surrounding', 'survivors', 'sustain',
        'sustainable', 'symphony', 'system', 'systems', 'take', 'teach',
        'teachers', 'teaching', 'team', 'technical', 'technology', 'teen',
        'teens', 'television', 'texas', 'theater', 'theatre', 'thousands',
        'three', 'thrive', 'throughout', 'time', 'times', 'today',
        'todays', 'together', 'tools', 'top', 'toward', 'tradition',
        'traditional', 'trained', 'training', 'transform', 'transitional',
        'transportation', 'treatment', 'treatments', 'two', 'underserved',
        'understand', 'understanding', 'unique', 'united', 'united states',
        'united way', 'university', 'upon', 'urban', 'us', 'use', 'used',
        'using', 'valley', 'value', 'values', 'variety', 'various', 'veterans',
        'vibrant', 'victims', 'violence', 'virginia', 'vision', 'visitors',
        'visual', 'vital', 'vocational', 'voice', 'volunteer', 'volunteers',
        'vulnerable', 'want', 'washington', 'water', 'way', 'ways', 'week',
        'welfare', 'well', 'wellbeing', 'wellness', 'west', 'western',
        'whole', 'whose', 'wide', 'wide range', 'wild', 'wildlife',
        'within', 'without', 'women', 'women children', 'womens',
        'word', 'work', 'worked', 'workforce', 'working', 'works',
        'workshops', 'world', 'worldclass', 'worlds', 'worldwide',
        'would', 'year', 'yearround', 'years', 'york', 'york city',
        'young', 'young people', 'youth', 'youth development'
        ]


def run_class_analysis(txt):
    """Classifies a charity description.

    :param txt: Charity description text
    """
    txt_no_punctuation = utils.remove_punctuation(txt)

    stop_words = stopwords.words('english')

    def clean_txt(x): return ' '.join(
        [word for word in x.split() if word not in (stop_words)])
    txt_clean = clean_txt(txt_no_punctuation)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words=stop_words,
        ngram_range=(1, 2),
        min_df=0,
        max_df=1
    )
    X = vectorizer.fit_transform([txt_clean])
    txt_tfidf = pd.DataFrame(
        X.toarray(), columns=vectorizer.get_feature_names())

    features = feat

    df_all = pd.DataFrame(np.nan, index=[0], columns=features)
    data = df_all.fillna(txt_tfidf)
    data = data.fillna(0)

    with open('model_lr.pkl', 'rb') as f:
        model_lr = pickle.load(f)

    cat = model_lr.predict(data)

    category_dict = {
        0: 'Human Services',
        1: 'Health',
        2: 'Education',
        3: 'Arts, Culture, Humanities',
        4: 'Religion',
        5: 'Research and Public Policy',
        6: 'Community Development',
        7: 'Animals',
        8: 'Human and Civil Rights',
        9: 'Environment'
    }

    result = category_dict[cat[0]]
    return result


img = 'https://www.python.org/static/community_logos/python-logo-master-v3-TM.png'

st.set_page_config(
    layout='wide',
    initial_sidebar_state='expanded',
    page_title='Intro',
    page_icon=img,
)

_, col2, _ = st.columns(3)
col2.image(img, caption='', width=300)
col2.markdown("""<h1 style='text-align: center; color: grey;\
    '>Charity Classification</h1>""", unsafe_allow_html=True)

st.markdown("""<h4 style='text-align: center; color: black;'>\
    This app classifies the charities description \
        into 10 categories</h4>""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])

col1.text('Categories')
col1.markdown(
    """
1. Animals
2. Arts, Culture, Humanities
3. Community Development
4. Education
5. Environment
6. Health
7. Human Services
8.  Human and Civil Rights
9.  Religion
10. Research and Public Policy
"""
)

col2.text('Charity Description')
txt = col2.text_area('Chariry Description',
                     ''' Working with Oregonians to enhance our quality\
 of life by building livable urban and rural\
 communities, protecting family farms\
 and forests, and conserving natural areas.
''',
                     height=200,
                     label_visibility='collapsed'
                     )

col3.text('Class')
if col2.button('Click!', type='primary'):
    col3.write(f'### {run_class_analysis(txt)}')


col2.markdown('##### Examples of charity description')
col2.markdown('''
1. *Serving the metropolitan Baltimore area and the state of Maryland,\
    the mission of WYPR Your Public Radio is to broadcast programs of\
        intellectual integrity and cultural merit which enrich the minds\
            and spirits of our listeners and ultimately strengthen the\
                communities we serve. WYPR adheres to the highest\
                    standards of journalistic and artistic excellence.\
                        It delivers educational, informational, cultural, and\
                         entertainment programming as a public services to the\
                             broadest possible audience.*
 '''
              )
col2.text('Category: Arts, Culture, Humanities')

col2.markdown('''


''')

col2.markdown('''
2. *VSS Catholic Communications is dedicated to answering the call\
    of the late Holy Father, Blessed John Paul II, for a New\
        Evangelization. Our charism in the mission of\
            evangelization is to broadly employ broadcast\
                media to transmit the Gospel of Jesus Christ\
                    with the fullness of the living Tradition\
                        as preserved, revered and proclaimed\
                            by the Catholic Church for more\
                                than 2,000 years. Everyone\
                                    involved recognizes this\
                                        apostolate as an\
                                            extraordinary\
                                                gift, which\
                                                     we\
                                                        receive\
                                                            anew\
                                                                each\
                                                                     day.*
 '''
              )
col2.text('Category: Religion')
