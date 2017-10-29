import os
import json
from csv import DictReader, DictWriter

import re
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from numpy import array
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import *
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer
SEED = 5


class ItemSelector(BaseEstimator, TransformerMixin):

    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


class NegativeWordSelector(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        return data


class NegativeWordTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.negators= ["never","neither","nobody","no","none","nor" ,"nothing","nowhere"]
        self.negative_words = ["abnormal","abolish","abominable","abomination","abort","aborted","abrade","abrasive","abrupt","abruptly","abscond","absence","absent-minded","absentee","absurd","absurdity","absurdly","abuse","abused","abuses","abusive","abysmal","abysmally","abyss","accidental","accost","accursed","accusation","accusations","accuse","accuses","accusing","acerbate","acerbic","ache","ached","aches","aching","acrimonious","acrimony","adamant","adamantly","addict","addicted","addicts","admonition","adulterate","adulterated","adversarial","adversary","adverse","adversity","afflict","affliction","affront","afraid","aggravate","aggravating","aggravation","aggression","aggressive","aggrieve","aggrieved","aghast","agonies","agonize","agonizing","agonizingly","agony","ail","ailing","ailment","aimless","alarm","alarmed","alarming","alienate","alienated","alienation","allegation","allegations","allege","allergic","allergies","allergy","aloof","altercation","ambiguity","ambiguous","ambivalence","ambivalent","ambush","amiss","amputate","anarchist","anarchy","anemic","anger","angrily","angry","anguish","animosity","annihilate","annihilation","annoy","annoyance","annoyances","annoyed","annoying","annoyingly","annoys","anomaly","antagonism","antagonist","antagonistic","antagonize","anti-","anti-social","anti-white","antiquated","anxieties","anxiety","anxious","anxiously","apathetic","apathy","apocalypse","apocalyptic","appal","appall","appalled","appalling","appallingly","apprehension","apprehensions","apprehensive","apprehensively","arbitrary","arcane","archaic","arduous","argumentative","arrogance","arrogant","ashamed","asinine","assail","assassin","assassinate","assault","astray","atrocious","atrocities","atrocity","atrophy","attack","attacks","audacious","audaciously","audacity","authoritarian","autocrat","avarice","avaricious","avenge","averse","aversion","awful","awfully","awfulness","awkward","awkwardness","ax","babble","backward","backwood","backwoods","bad","badly","baffle","baffled","baffling","bait","balk","banal","bane","banish","banishment","bankrupt","barbarian","barbaric","barbarity","barren","baseless","bash","bashed","bashful","bashing","bastard","bastards","battered","battering","bearish","befoul","beg","beggar","begging","belabor","belated","beleaguer","belie","belittle","belittled","belittling","belligerent","bemoan","bemoaning","bemused","bent","berate","bereave","bereavement","bereft","berserk","beseech","beset","besiege","bestial","betray","betrayal","betrayals","betraying","betrays","beware","bewilder","bewildered","bewildering","bewilderingly","bewilderment","bewitch","bias","biased","biases","bicker","bickering","bigotry","bitch","bitchy","biting","bitingly","bitter","bitterly","bitterness","bizarre","blab","blackmail","blah","blame","bland","blasphemy","blasted","blatant","blatantly","blather","bleak","bleakly","bleakness","bleed","bleeding","bleeds","blemish","blind","blinding","blindingly","blindside","blister","blistering","bloated","blockhead","bloodshed","bloodthirsty","bloody","blow","blunder","blunders","blunt","blur","blurred","blurry","blurs","blurt","boastful","boggle","bogus","boil","boiling","boisterous","bomb","bombard","bombastic","bondage","bonkers","bore","bored","boredom","bores","boring","botch","bother","bothered","bothering","bothers","bothersome","braggart","brainless","brainwash","brash","brashly","brashness","brat","bravado","brazen","brazenly","breach","break","break-up","break-ups","breakdown","breaking","breaks","breakup","breakups","bribery","brimstone","brittle","broke","broken","brood","browbeat","bruise","bruised","brusque","brutal","brutality","brutalize","brutally","brute","brutish","bs","buckle","bug","buggy","bugs","bulky","bullies","bullshit","bully","bullying","bum","bump","bumped","bumping","bumps","bumpy","bungle","bungler","bungling","bunk","burden","burdensome","burn","burned","burning","burns","bust","busts","busybody","butcher","butchery","buzzing","cackle","calamitous","callous","cancer","cannibal","cannibalize","capitulate","capricious","capriciously","careless","caricature","carnage","carp","cartoonish","cash-strapped","castigate","casualty","catastrophe","catastrophes","catastrophic","caustic","cautionary","cave","chagrin","challenging","chaos","chaotic","chastise","chatter","chatterbox","cheap","cheapen","cheaply","cheat","cheated","cheater","cheating","cheats","cheesy","chide","childish","chill","chilly","chintzy","choke","choppy","chore","chronic","chunky","clamor","clash","cliche","cliched","clique","cloud","cloudy","clueless","clumsy","clunky","coarse","cocky","coerce","coercion","cold","coldly","collapse","combative","comical","commiserate","commonplace","commotion","complacent","complain","complained","complaining","complains","complaint","complaints","complex","complicated","complication","complicit","compulsion","compulsive","concede","conceit","conceited","concen","concern","concerned","concerns","condemn","condemnation","condemned","condemns","condescend","condescending","condescension","confess","confession","confessions","confined","conflict","conflicted","conflicting","conflicts","confound","confounding","confront","confrontation","confrontational","confuse","confused","confuses","confusing","confusion","confusions","cons","conservative","conspicuous","conspicuously","conspiracies","conspiracy","conspirator","conspiratorial","conspire","consternation","contagious","contaminate","contaminated","contempt","contemptible","contemptuous","contemptuously","contend","contention","contentious","contort","contortions","contradict","contradiction","contradictory","contrive","contrived","controversial","controversy","convoluted","corrupt","corrupted","corrupting","corruption","costly","counter-productive","covetous","coward","cowardly","crabby","crack","cracked","cracks","craftily","crafty","cramp","cramped","cranky","crap","crappy","craps","crash","crashed","crashes","crashing","crass","craven","craze","craziness","crazy","creak","creaking","creaks","credulous","creep","creeping","creeps","creepy","crept","crime","criminal","cringe","cringed","cringes","cripple","crippled","crippling","crisis","critic","critical","criticism","criticisms","criticize","criticized","criticizing","critics","crook","crooked","crooks","crowded","crude","cruel","cruelly","cruelties","cruelty","crumble","crumbling","crummy","crumple","crumpled","crush","crushed","crushing","cry","culprit","cumbersome","curse","cursed","curses","curt","cuss","cussed","cutthroat","cynical","cynicism","damage","damaged","damages","damaging","damn","damnation","damned","damning","damper","danger","dangerous","dark","darken","darkened","darker","darkness","dastard","dastardly","daunt","daunting","dauntingly","daze","dazed","dead","deadbeat","deadly","deaf","dearth","death","debacle","debase","debatable","debauch","debaucher","debauchery","debilitate","debilitating","debt","debts","decadence","decadent","decay","decayed","deceit","deceitful","deceive","deceiver","deceiving","deception","deceptive","deceptively","decline","declines","declining","decrement","decrepit","defect","defects","defensive","defiance","defiant","defiantly","deficiencies","deficiency","deficient","defile","deform","deformed","defunct","defy","degenerate","degeneration","degradation","degrade","degrading","dehumanization","deject","dejected","dejectedly","dejection","delay","delayed","delinquent","delirious","delirium","delude","deluded","deluge","delusion","delusional","delusions","demean","demeaning","demise","demolish","demon","demonic","denial","denied","denies","denigrate","denounce","dense","dent","dented","dents","deny","denying","deplete","deplorable","deplore","deprave","depraved","depress","depressed","depressing","depressingly","depression","deprive","deprived","deride","derisive","derogatory","desert","desertion","desolate","desolation","despair","desperate","desperately","desperation","despicable","despise","despised","despondence","despondent","destitute","destroy","destroyer","destruction","destructive","deter","deteriorate","deteriorating","deterioration","detest","detestable","detested","detests","detract","detracted","detracting","detraction","detracts","detriment","detrimental","devastate","devastated","devastating","devastatingly","devastation","deviate","deviation","devil","devilish","devious","deviously","deviousness","devoid","diabolic","diabolical","diabolically","diametrically","diatribe","diatribes","dick","dictator","die","die-hard","died","dies","difficult","difficulties","difficulty","dilapidated","dilemma","dim","din","ding","dings","dinky","dire","dirt","dirty","disable","disabled","disadvantage","disadvantageous","disaffect","disaffected","disaffirm","disagree","disagreeable","disagreed","disagreement","disagrees","disallow","disappoint","disappointed","disappointing","disappointingly","disappointment","disappointments","disappoints","disapprobation","disapproval","disapprove","disapproving","disarm","disarray","disaster","disasterous","disastrous","disastrously","disbelief","disbelieve","disclaim","discombobulate","discomfort","disconcert","disconcerted","disconcerting","disconcertingly","discontent","discontented","discord","discordant","discourage","discouraging","discouragingly","discourteous","discredit","discrimination","disdain","disdainful","disgrace","disgraced","disgraceful","disgruntle","disgruntled","disgust","disgusted","disgustedly","disgusting","disgustingly","dishearten","disheartening","dishearteningly","dishonest","dishonor","disillusion","disillusioned","disillusionment","disingenuous","disintegrate","disintegrated","disintegration","disinterest","disinterested","dislike","disliked","dislikes","disliking","disloyal","dismal","dismally","dismay","dismayed","dismaying","dismayingly","dismissive","disobedience","disobey","disorder","disorient","disoriented","disown","dispirit","dispirited","displace","displaced","displease","displeased","displeasing","displeasure","disprove","disputable","dispute","disputed","disquiet","disquieting","disregard","disreputable","disrespect","disrespectful","disrespecting","disrupt","disruption","diss","dissapointed","dissappointed","dissatisfaction","dissatisfied","dissatisfy","dissatisfying","dissemble","dissent","disservice","dissing","dissolute","dissuade","dissuasive","distaste","distasteful","distort","distorted","distortion","distorts","distract","distracting","distraction","distraught","distress","distressed","distressing","distressingly","distrust","distrustful","disturb","disturbance","disturbed","disturbing","disturbingly","divergent","divisive","dizzy","doddering","dogged","dogmatic","domineer","domineering","doom","doomed","doomsday","dope","doubt","doubtful","doubts","douchebag","downbeat","downcast","downer","downfall","downgrade","downhill","downside","downsides","downturn","drab","draconian","drag","dragged","dragging","drags","drain","drained","draining","drains","drastic","drastically","drawback","drawbacks","dread","dreadful","dreadfully","dreary","dripping","drippy","drips","drones","droop","drop-out","dropout","dropouts","drought","drowning","drunk","drunkard","drunken","dubious","dud","dull","dullard","dumb","dumbfound","dump","dumped","dumping","dumps","dunce","dungeon","dungeons","dupe","dust","dusty","dwindling","dying","eccentric","eccentricity","egocentric","egomania","egotism","egotistical","egregious","elimination","emasculate","embarrass","embarrassing","embarrassingly","embarrassment","embroil","embroiled","emergency","emphatic","emphatically","emptiness","encroach","encroachment","endanger","enemies","enemy","engulf","enmity","enrage","enraged","enraging","enslave","entangle","entanglement","entrap","entrapment","envious","epidemic","equivocal","erase","erode","erosion","err","errant","erratic","erratically","erroneous","error","errors","escapade","eschew","estranged","evade","evasion","evasive","evil","evils","eviscerate","exacerbate","exaggerate","exaggeration","exasperate","exasperated","exasperating","exasperation","excessive","excessively","exclusion","excruciating","excruciatingly","excuse","excuses","exhaust","exhausted","exhaustion","exile","exorbitant","exorbitantly","expel","expensive","expire","expired","explode","exploit","exploitation","explosive","expunge","exterminate","extermination","extinguish","extort","extortion","extraneous","extravagance","extravagant","extremism","extremist","extremists","fabricate","fabrication","facetious","fail","failed","failing","fails","failure","failures","faint","faithless","fake","fall","fallacy","fallen","falling","falls","false","falsehood","falsely","falter","faltered","famished","fanatic","fanatical","fanatically","fanaticism","fanatics","fanciful","far-fetched","farce","farcical","farfetched","fascism","fascist","fastidious","fat","fatal","fatalistic","fatally","fateful","fatigue","fatigued","fatty","fatuous","fault","faults","faulty","faze","fear","fearful","fearfully","fears","fearsome","feeble","feign","feint","fell","felon","ferociously","ferocity","fever","feverish","fiasco","fib","fibber","fickle","fiction","fictional","fictitious","fidget","fiend","fiendish","fierce","figurehead","filth","filthy","fist","flabbergast","flabbergasted","flagging","flair","flak","flake","flaky","flare","flares","flat-out","flaunt","flaw","flawed","flaws","flee","fleeing","flees","fleeting","flicker","flickering","flickers","flighty","flimsy","flirt","floored","flounder","floundering","fluster","foe","fool","fooled","foolish","foolishly","foolishness","forbid","forbidden","forbidding","forceful","foreboding","forfeit","forged","forgetful","forlorn","forsake","forsaken","foul","fracture","fragile","fragmented","frail","frantic","frantically","fraud","fraught","frazzle","frazzled","freak","freaking","freakish","freaks","freeze","freezes","freezing","frenetic","frenetically","frenzied","frenzy","fret","fretful","friction","fried","friggin","fright","frighten","frightening","frighteningly","frightful","frightfully","frigid","frost","frown","froze","frozen","fruitless","frustrate","frustrated","frustrates","frustrating","frustration","frustrations","fuck","fucking","fudge","fugitive","full-blown","fumble","fume","fumes","funky","funny","furious","furiously","fury","fuss","fussy","futile","futility","fuzzy","gaff","gaffe","gall","gangster","gape","garbage","garish","gasp","gaudy","gawk","gawky","geezer","genocide","ghastly","ghetto","giddy","gimmick","gimmicks","gimmicky","glare","glaringly","glib","glitch","glitches","gloom","gloomy","glower","glum","glut","gnawing","goad","goof","goofy","goon","gossip","graceless","graft","grainy","grapple","grate","grating","gravely","greasy","greed","greedy","grief","grievance","grievances","grieve","grieving","grievous","grievously","grim","grimace","grind","gripe","gripes","grisly","gritty","gross","grossly","grotesque","grouch","grouchy","groundless","grouse","growl","grudge","grudges","grudging","grudgingly","gruesome","gruesomely","gruff","grumble","grumpier","grumpiest","grumpy","guile","guilt","guiltily","guilty","gullible","gutless","gutter","hack","hacks","haggard","halfhearted","halfheartedly","hallucinate","hallucination","hamper","hampered","handicapped","hang","hangs","haphazard","hapless","harass","harassed","harasses","harassment","harboring","harbors","hard","hard-hit","hardball","harden","hardened","hardheaded","hardship","hardships","harm","harmed","harmful","harms","harridan","harried","harrow","harsh","harshly","hassle","hassled","hassles","haste","hastily","hasty","hate","hated","hateful","hater","haters","hates","hating","hatred","haughty","haunt","haunting","havoc","haywire","hazard","hazardous","haze","hazy","headache","headaches","heartbreaker","heartbreaking","heartless","heavy-handed","heck","heckle","hectic","hedge","hedonistic","hefty","heinous","hell","hell-bent","hells","helpless","helplessly","helplessness","heresy","heretic","heretical","hesitant","hideous","hideously","hideousness","high-priced","hinder","hindrance","hiss","hissed","hissing","ho-hum","hoax","hogs","hollow","hoodwink","hooligan","hopeless","hopelessly","hopelessness","horde","horrendous","horrendously","horrible","horrid","horrific","horrified","horrifies","horrify","horrifying","hostage","hostile","hostility","hothead","hotheaded","hubris","huckster","hum","humid","humiliate","humiliating","humiliation","humming","hung","hurt","hurtful","hurting","hurts","hustler","hype","hypocrisy","hypocrite","hypocrites","hypocritical","hysteria","hysteric","hysterical","hysterically","hysterics","idiocy","idiot","idiotic","idiotically","idiots","idle","ignorance","ignorant","ignore","ill-advised","ill-conceived","ill-defined","ill-fated","ill-natured","ill-tempered","illegal","illegally","illegitimate","illicit","illiterate","illness","illogic","illogical","illogically","illusion","illusions","illusory","imaginary","imbalance","imbecile","immature","immoral","immorality","impair","impaired","impatience","impatient","impatiently","impeach","impediment","impending","imperfect","imperfection","imperfections","imperialist","imperious","imperiously","impersonal","impertinent","impetuous","implacable","implausible","implicate","implication","implode","impolite","impose","imposing","impossible","impossibly","impotent","impoverish","impoverished","imprison","imprisonment","improbable","improbably","improper","improperly","impulsive","impulsively","impunity","inability","inaccuracies","inaccurate","inaccurately","inaction","inactive","inadequacy","inadequate","inane","inanely","inappropriate","inappropriately","incapable","incendiary","incense","incessant","incessantly","incite","inclement","incoherence","incoherent","incomparable","incompatible","incompetence","incompetent","incompetently","incomplete","incomprehensible","inconceivable","incongruous","incongruously","inconsequent","inconsequential","inconsiderate","inconsistencies","inconsistency","inconsistent","inconvenience","incorrect","incorrectly","incorrigible","incredulous","incredulously","indecency","indecent","indecisive","indeterminate","indifference","indifferent","indignant","indignantly","indignity","indiscretion","indistinguishable","indoctrinate","indolent","indulge","ineffective","ineffectual","inelegant","inept","ineptitude","ineptly","inescapable","inevitable","inevitably","inexcusable","inexcusably","inexorable","inexorably","inexperience","inexperienced","inextricably","infamous","infamously","infamy","infected","infection","inferior","inferiority","infest","infested","infidel","infirm","inflame","inflated","inflict","infuriate","infuriated","infuriating","infuriatingly","inhibit","inhibition","inhospitable","inhuman","inhumanity","iniquitous","iniquity","injure","injurious","injury","injustice","injustices","innuendo","inopportune","inordinate","insane","insanely","insanity","insatiable","insecure","insecurity","insensitive","insidious","insidiously","insignificant","insincere","insincerely","insincerity","insinuate","insinuating","insinuation","insolent","insouciance","instability","instigate","insubordinate","insubstantial","insufferable","insufficient","insular","insult","insulted","insulting","insultingly","insults","insurmountable","insurrection","intense","interfere","interference","interferes","intermittent","interrupt","interruption","interruptions","intimidate","intimidating","intimidation","intolerable","intolerance","intoxicate","intrude","intrusion","intrusive","inundate","inundated","invader","invasive","invective","inveigle","invisible","involuntarily","involuntary","irascible","irate","ire","irk","irked","irking","irks","ironic","ironical","ironically","ironies","irony","irrational","irrationalities","irrationality","irregular","irrelevant","irreparable","irrepressible","irresponsible","irreversible","irritable","irritate","irritated","irritating","irritation","isolate","isolated","isolation","issue","issues","itch","itching","itchy","jabber","jaded","jagged","jam","jarring","jealous","jealously","jealousy","jeer","jeopardize","jeopardy","jerk","jerky","jitter","jitters","jittery","jobless","joke","joker","jolt","jumpy","junk","junky","junkyard","kill","killed","killer","killing","kills","knife","knock","kook","kooky","lack","lackadaisical","lacked","lackey","lackeys","lacking","lackluster","lacks","laconic","lag","lagging","lags","lambast","lambaste","lame","lament","lamentable","languid","languish","lanky","lapse","lapsed","lapses","lascivious","last-ditch","laughable","laughably","lawless","lawlessness","layoff","lazy","leak","leaking","leaks","lech","lecher","lecherous","lechery","leech","leer","leery","lemon","lengthy","letch","lethal","lethargic","lewd","liability","liable","liar","liars","licentious","lie","lied","lier","lies","life-threatening","lifeless","limit","limitation","limitations","limited","limits","limp","listless","little-known","loath","loathe","loathing","loathsome","lone","loneliness","lonely","loner","lonesome","long-time","long-winded","longing","longingly","loophole","loose","loot","lorn","lose","loser","losers","loses","losing","loss","losses","lost","loud","louder","lousy","loveless","lovelorn","lowly","ludicrous","ludicrously","lugubrious","lukewarm","lull","lumpy","lunatic","lurch","lure","lurid","lurk","lurking","lying","macabre","mad","madden","maddening","maddeningly","madly","madman","madness","maladjusted","maladjustment","malaise","malcontent","malevolence","malevolent","malice","malicious","maliciously","malign","mangle","mangled","mania","maniac","maniacal","manic","manipulate","manipulation","manipulative","mar","marginal","marginally","mashed","massacre","massacres","matte","mawkish","mawkishness","meager","meaningless","meanness","measly","meddle","mediocre","mediocrity","melancholy","melodramatic","meltdown","menace","menacing","menacingly","menial","merciless","mercilessly","mess","messed","messes","messing","messy","midget","miff","mindless","mindlessly","mire","misbegotten","misbehave","miscalculate","miscalculation","mischief","mischievous","mischievously","misconception","misconceptions","misdirection","miser","miserable","miserably","miseries","miserly","misery","misfit","misfortune","misgiving","misgivings","misguide","misguided","mishandle","mishap","misinterpret","misjudgment","mislead","misleading","mispronounce","mispronounced","mispronounces","misrepresent","misrepresentation","miss","missed","misses","mist","mistake","mistaken","mistakenly","mistakes","mistress","mistrust","mistrustful","mists","misunderstand","misunderstanding","misunderstandings","misunderstood","misuse","moan","mobster","mock","mocked","mockery","mocking","mockingly","mocks","molest","molestation","monotonous","monotony","monster","monstrosity","monstrous","monstrously","moody","moot","mope","morbid","mordant","mordantly","moribund","moron","moronic","morons","mortified","motionless","motley","mourn","mournful","muddle","muddy","mundane","murder","murderer","murderous","murky","mushy","musty","mysterious","mysteriously","mystery","mystify","myth","nag","nagging","naive","nastiness","nasty","naughty","nauseate","nauseating","nauseatingly","nebulous","needless","needlessly","needy","nefarious","negate","negative","negatives","negativity","neglect","neglected","negligence","negligent","nemesis","nervous","nervously","nervousness","neurotic","neurotically","niggle","niggles","nightmare","nightmarish","nightmarishly","nitpick","noise","noises","noisy","nonexistent","nonsense","notoriety","notorious","notoriously","noxious","nuisance","numb","obese","object","objection","objectionable","objections","oblique","obliterate","obliterated","oblivious","obnoxious","obnoxiously","obscene","obscenely","obscenity","obscure","obscured","obscures","obscurity","obsess","obsessive","obsessively","obsolete","obstacle","obstruct","obstructed","obstructing","obstruction","obtrusive","obtuse","odd","odder","oddest","oddities","oddity","oddly","odor","offence","offend","offender","offending","offensive","offensively","offensiveness","officious","ominous","ominously","omission","omit","one-sided","onslaught","opinionated","opponent","opportunistic","oppose","opposition","oppress","oppression","oppressive","oppressively","oppressors","ordeal","orphan","outbreak","outburst","outbursts","outcast","outlaw","outrage","outraged","outrageous","outrageously","outrageousness","outsider","over-hyped","overact","overacted","overbearing","overbearingly","overblown","overdo","overdone","overdue","overheat","overkill","overloaded","overlook","overpaid","overplay","overpower","overpriced","overrated","overreach","overrun","overshadow","oversight","oversimplification","oversimplified","oversize","overstate","overstated","overstatement","overthrow","overturn","overweight","overwhelm","overwhelmed","overwhelming","overwhelmingly","overwhelms","overzealous","pain","painful","painfull","painfully","pains","pale","pales","paltry","pan","pandemonium","pander","pandering","panders","panic","panick","panicked","panicking","panicky","paradoxical","paradoxically","paralize","paralyzed","paranoia","paranoid","parasite","pariah","parody","partisan","partisans","passe","passive","pathetic","pathetically","patronize","paucity","payback","peculiar","peculiarly","pedantic","peeled","penalty","perfunctory","peril","perilous","perilously","perish","perplex","perplexed","perplexing","persecute","persecution","perturb","perturbed","pervasive","perverse","perversely","perversion","pervert","perverted","pessimism","pessimistic","pest","petrified","petty","phobia","phobic","phony","picket","pickets","picky","pig","pigs","pillage","pimple","pinch","pitiful","pitifully","pittance","pity","plagiarize","plague","plaything","plea","pleas","plight","plot","ploy","plunder","pointless","pointlessly","poison","poisonous","pollute","pompous","poor","poorer","poorly","posturing","pout","poverty","powerless","pratfall","prattle","precarious","precariously","precipitous","predatory","predicament","prejudge","prejudice","prejudices","premeditated","preposterous","preposterously","pretend","pretense","pretentious","pretentiously","pricey","prick","prideful","prik","primitive","prison","prisoner","problem","problematic","problems","procrastination","profane","profanity","prohibit","propaganda","prosecute","protest","protesting","protests","protracted","provocation","provocative","provoke","pry","pugnacious","punch","punish","punishable","punk","puny","puppet","puppets","puzzled","puzzling","quack","qualm","qualms","quandary","quarrel","quarrels","quash","questionable","quibble","quibbles","rabid","racism","racist","racy","radical","radically","radicals","rage","ragged","raging","rail","rampage","rampant","ramshackle","rancor","randomly","rant","ranted","ranting","rants","rape","raped","raping","rascal","rash","rattle","rattles","ravage","raving","reactionary","rebellious","rebuff","rebuke","recant","reckless","recklessly","recklessness","recoil","redundant","refusal","refuse","refused","refuses","refusing","refuting","regress","regression","regressive","regret","regretful","regretfully","regrets","regrettable","regrettably","regretted","reject","rejected","rejecting","rejection","rejects","relentless","relentlessly","relentlessness","reluctance","reluctant","reluctantly","remorse","remorseful","renounce","repel","repetitive","reprehensible","repress","repression","repressive","reprimand","reproach","repugn","repugnant","repulse","repulsed","repulsing","repulsive","repulsively","resent","resentful","resentment","resignation","resigned","resistance","restless","restlessness","restrict","restricted","restriction","restrictive","retaliate","retard","retarded","reticent","retreat","retreated","revenge","revengeful","revert","revile","reviled","revoke","revolt","revolting","revoltingly","revulsion","rhetoric","rhetorical","ridicule","ridiculous","ridiculously","rife","rift","rifts","rigid","rile","riled","rip","rip-off","ripoff","ripped","risk","risks","risky","rival","rivalry","roadblocks","rocky","rogue","rollercoaster","rot","rotten","rough","rubbish","rude","rue","ruffian","ruffle","ruin","ruined","ruining","ruins","rumbling","rumor","rumors","rumours","rumple","run-down","runaway","rupture","rust","rusts","rusty","rut","ruthless","ruthlessness","ruts","sabotage","sack","sacrificed","sad","sadden","sadly","sadness","sag","sagged","sagging","sags","salacious","sanctimonious","sap","sarcasm","sarcastic","sardonic","sass","satirical","satirize","savage","savaged","savagery","savages","scam","scams","scandal","scandalous","scandals","scant","scapegoat","scar","scarce","scarcely","scare","scared","scarier","scariest","scarred","scars","scary","scathing","scathingly","scoff","scold","scolding","scorching","scorn","scornful","scoundrel","scowl","scramble","scrambled","scrambles","scrambling","scrap","scratch","scratched","scratches","scratchy","scream","screech","screw-up","screwed","screwed-up","screwy","scuff","scum","scummy","second-class","second-tier","secretive","seedy","seething","self-defeating","self-destructive","self-serving","selfish","selfishly","selfishness","senile","senseless","seriousness","sermonize","set-up","setback","setbacks","sever","severe","severity","shabby","shadowy","shady","shake","shaky","shallow","sham","shambles","shame","shameful","shameless","shamelessly","shark","sharply","shatter","shimmer","shimmy","shipwreck","shirk","shit","shiver","shock","shocked","shocking","shockingly","shoddy","short-lived","shortage","shortcoming","shortcomings","shortsighted","showdown","shrew","shriek","shrill","shroud","shrouded","shrug","shun","shunned","sick","sicken","sickening","sickeningly","sickly","sickness","sidetrack","sidetracked","siege","silly","simplistic","simplistically","sin","sinful","sinfully","sinister","sink","sinking","skeletons","skeptic","skeptical","skepticism","sketchy","skimpy","skinny","skittish","skulk","slack","slander","slap","slashing","slaughter","slaughtered","slave","slaves","sleazy","slime","slog","slogging","slogs","sloppily","sloppy","sloth","slothful","slow","slow-moving","slowed","slower","slowest","slowly","slug","sluggish","slump","slumping","slur","slut","sluts","sly","smack","smallish","smash","smear","smell","smelled","smelling","smells","smelly","smoke","smokescreen","smolder","smoldering","smother","smug","smugly","smut","smutty","snag","snagged","snags","snare","snarky","snarl","sneak","sneaky","sneer","sneering","snob","snobbish","snobby","snobs","snub","so-cal","sob","sober","solemn","somber","sore","sorely","sorrow","sorrowful","sorry","sour","spade","spank","spew","spewed","spewing","spews","spilling","spinster","spiritless","spite","spiteful","splatter","split","splitting","spoil","spoiled","spoils","spook","spooky","spoon-fed","spoon-feed","sporadic","spotty","spurn","sputter","squabble","squabbling","squander","squash","squeak","squeaky","squeal","squealing","squeals","squirm","stab","stagnant","stagnate","stain","stains","stale","stall","stalls","stammer","stampede","stark","starkly","startle","startling","startlingly","starve","static","steal","stealing","steals","steep","stench","stereotype","stereotypical","stereotypically","stern","stew","sticky","stiff","stifle","stifling","stigma","sting","stingy","stink","stinks","stodgy","stole","stolen","stooge","stooges","stormy","strain","strained","straining","strange","strangely","stranger","strangest","strangle","strenuous","stress","stresses","stressful","stricken","strict","strictly","strident","stridently","strife","strike","stringent","struck","struggle","struggled","struggles","struggling","strut","stubborn","stubbornly","stubbornness","stuck","stuffy","stumble","stumbled","stumbles","stump","stumped","stun","stunt","stunted","stupid","stupidest","stupidity","stupidly","stupor","stutter","stuttering","stutters","sty","stymied","sub-par","subdued","subjected","subjugate","subordinate","subservient","substandard","subtract","subversive","subvert","succumb","suck","sucked","sucker","sucks","sue","sued","sues","suffer","suffered","suffering","suffers","suffocate","sugar-coat","sugar-coated","suicidal","suicide","sulk","sullen","sully","sunder","sunk","sunken","superficial","superficiality","superficially","superfluous","superstition","suppress","suppression","surrender","susceptible","suspect","suspicion","suspicions","suspicious","suspiciously","swagger","swamped","sweaty","swelled","swelling","swindle","swipe","symptom","symptoms","syndrome","taboo","tacky","taint","tainted","tamper","tangle","tangled","tangles","tank","tanked","tanks","tantrum","tarnish","tarnished","tattered","taunt","taunting","taunts","taut","tawdry","taxing","tease","tedious","tediously","temerity","temper","tempest","temptation","tenderness","tense","tension","tentative","tentatively","tenuous","tepid","terrible","terribly","terror","terrorism","terrorize","testy","thankless","thicker","thirst","thorny","thoughtless","thoughtlessly","thrash","threat","threaten","threatening","threats","threesome","throb","throbbing","throttle","thug","thwart","time-consuming","timid","timidity","tingling","tired","tiresome","tiring","toil","toll","topple","torment","tormented","torrent","tortuous","torture","tortured","tortures","torturing","torturous","torturously","totalitarian","touchy","toughness","tout","touted","touts","toxic","tragedy","tragic","tragically","traitor","traitorous","tramp","trample","transgress","transgression","trap","trapped","trash","trashed","trashy","trauma","traumatic","traumatize","traumatized","travesty","treacherous","treachery","treason","trick","tricked","trickery","tricky","trivial","trivialize","trouble","troubled","troublemaker","troubles","troublesome","troubling","truant","tumble","tumbled","tumbles","tumultuous","turbulent","turmoil","twist","twisted","twists","tyrannical","tyranny","tyrant","ugh","ugliest","ugliness","ugly","ulterior","ultimatum","unable","unacceptable","unappealing","unattractive","unauthentic","unavailable","unbearable","unbelievable","unbelievably","uncaring","uncertain","unclean","unclear","uncomfortable","uncomfortably","uncompromising","uncompromisingly","uncontrolled","unconvincing","unconvincingly","uncooperative","uncouth","uncreative","undecided","undefined","undependable","undercut","undercuts","underdog","underestimate","underlings","undermine","undermined","undermines","undermining","undesirable","undid","undocumented","undone","unease","uneasily","uneasiness","uneasy","unemployed","unethical","uneven","uneventful","unexpected","unexpectedly","unexplained","unfairly","unfaithful","unfamiliar","unfeeling","unfinished","unfit","unforeseen","unforgiving","unfortunate","unfortunately","unfounded","unfriendly","unfulfilled","unhappily","unhappiness","unhappy","unhealthy","unimaginable","unimportant","uninformed","uninsured","unintelligible","unjust","unjustifiable","unjustified","unjustly","unkind","unknown","unlawful","unleash","unlicensed","unlikely","unlucky","unmoved","unnatural","unnecessary","unneeded","unnerve","unnerved","unnerving","unnoticed","unobserved","unorthodox","unpleasant","unpopular","unpredictable","unprepared","unprove","unproven","unqualified","unravel","unraveled","unreachable","unrealistic","unreasonable","unreasonably","unrelenting","unrelentingly","unreliable","unresolved","unrest","unruly","unsatisfactory","unsavory","unscrupulous","unsecure","unsettle","unsettled","unsettling","unsettlingly","unskilled","unsophisticated","unsound","unspeakable","unspecified","unstable","unsuccessful","unsuccessfully","unsupportive","unsure","unsuspecting","unthinkable","untimely","untouched","untrue","unusable","unusual","unusually","unwanted","unwarranted","unwatchable","unwelcome","unwieldy","unwilling","unwillingly","unwillingness","unwise","unwisely","unworkable","unworthy","upheaval","uprising","uproar","uproarious","uproariously","uproot","upset","upsets","upsetting","urgent","useless","usurp","usurper","utterly","vagrant","vague","vain","vainly","vanity","vehement","vehemently","vengeance","vengeful","venom","venomous","vent","vestiges","vex","vexing","vibrate","vibrates","vibrating","vibration","vice","vicious","viciously","viciousness","victimize","vile","villainous","villains","villian","vindictive","violate","violation","violator","violent","violently","viper","virulent","virus","volatile","volatility","vomit","vomiting","vomits","vulgar","vulnerable","wack","wail","wallow","wane","waning","wanton","warned","warning","warp","warped","wary","washed-out","waste","wasted","wasteful","wasting","watered-down","wayward","weak","weaken","weakening","weaker","weakness","weaknesses","weariness","wearisome","weary","wedge","weed","weep","weird","weirdly","whimper","whine","whining","whiny","whips","whore","whores","wicked","wickedly","wickedness","wild","wildly","wilt","wily","wimpy","wince","wobble","wobbles","woe","woeful","woefully","womanizer","womanizing","worn","worried","worriedly","worries","worrisome","worry","worrying","worse","worsen","worsening","worst","worthless","wound","wounds","wrath","wreak","wreaked","wreaks","wreck","wrest","wrestle","wretch","wretched","wrinkle","wrinkled","wrinkles","wrong","wrongful","wrongly","wrought","yawn","zap","zapped","zealot","zealous","zealously","zombie"]
        self.negative_prefix = ["dis","im","in","non","un"]

    def fit(self, examples):
        return self

    def transform(self, examples):
        negative_score = -1
        features = np.zeros((len(examples), 1))
        j = 0
        for ex in examples:
            words = ex.split(" ")
            features[j, 0]=0
            for i in range(len(words)):
                if words[i] in self.negative_words:
                    if words[i-1] not in self.negators:
                        features[j,0]+=negative_score           
                elif words[i][:2] in self.negative_prefix or words[i][:3] in self.negative_prefix:
                        features[j,0]+=negative_score
            j += 1
        return features

class QuoteTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, examples):
        return self

    def transform(self, examples):
        features = np.zeros((len(examples), 1))
        j = 0
        for ex in examples:
            words = ex.split(" ")
            features[j, 0]=0
            for i in range(len(words)):
                if re.search('\"',words[i]):
                    features[j]+=1
            j += 1
        return features


class AdverbTransform(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, examples):
        return self

    def transform(self, examples):
        features = np.zeros((len(examples), 1))
        j = 0
        for ex in examples:
            features[j]=len(re.findall(r'\w+ly', ex))
            j += 1
        return features

class PunctutationTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, examples):
        return self

    def transform(self, examples):
        features = np.zeros((len(examples), 3))
        j = 0
        for ex in examples:
            words = ex.split(" ")
            features[j, 0]=0
            features[j, 1]=0
            features[j, 2]=0
            for i in range(len(words)):
                if re.search('\. \. \.',words[i]):
                    features[j,0]+=1
                if re.search('! ! !',words[i]):
                    features[j,1]+=1
                if re.search(r'\? \? \?',words[i]):
                    features[j,2]+=1
            j += 1
        return features

class RateTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, examples):
        return self

    def transform(self, examples):
        features = np.zeros((len(examples), 1))
        j = 0
        for ex in examples:
            features[j, 0]=0
            matches = re.findall(r'\d\/\d\d',ex)
            if len(matches) > 0:
                rating = matches[len(matches)-1]
                num = re.split('\/',rating)
                if  len(num[0]) >0 and self.is_number(num[0]) and self.is_number(num[1]): 
                    rating=int(num[0])/int(num[1])
                    if rating < 0.5:
                        features[j,0] = -1
                    else:
                        features[j,0] = 1
            j += 1
        return features


    def is_number(self,s):
        try:
            float(s)
            return True
        except ValueError:
            return False

class TextLengthTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, examples):
        return self

    def transform(self, examples):
        features = np.zeros((len(examples), 1))
        i = 0
        for ex in examples:
            features[i, 0] = len(ex)
            i += 1
        return features

# TODO: Add custom feature transformers for the movie review data


class Featurizer:
    def __init__(self):
        # To add new features, just add a new pipeline to the feature union
        # The ItemSelector is used to select certain pieces of the input data
        # In this case, we are selecting the plaintext of the input data

        # TODO: Add any new feature transformers or other features to the FeatureUnion
        self.all_features = FeatureUnion([
            
            ('quotes', Pipeline([
                ('quotes_transformer',QuoteTransformer()),
                ('scale',Normalizer())              
            ])),
            ('char_ngram_pipeline', Pipeline([
                ('char_ngram',CountVectorizer(analyzer='char', ngram_range=(2, 2))),
                ('scale',Normalizer())              
            ])),
            ('text_stats', Pipeline([
                ('selector', CountVectorizer(stop_words='english',ngram_range=(1,2))),
                ('text_length',TfidfTransformer())
            ])),
            ('negative_words', Pipeline([
                ('negative_word_transformer',NegativeWordTransformer()),
                ('scale',Normalizer())                
            ]))
            ])

            
    def train_feature(self, examples):
        train_features = self.all_features.fit_transform(examples)
        return train_features

    def test_feature(self, examples):
        test_features= self.all_features.transform(examples)
        return test_features

if __name__ == "__main__":

    # Read in data

    dataset_x = []
    dataset_y = []
    i=0
    with open('hotelNegT-train.txt',encoding="utf8") as f:
        for data in f.readlines():    
            s = (data.split('\t',1)[1].lower())
            s=re.sub('\d', '', s)
            s=re.sub(r'\(', '', s)
            s=re.sub(r'\)', '', s)
            dataset_x.append(s)
            dataset_y.append(-1)
            i+=1
    with open('hotelPosT-train.txt',encoding="utf8") as f:
        for data in f.readlines():    
            s = (data.split('\t',1)[1].lower())
            s=re.sub('\d', '', s)
            s=re.sub(r'\(', '', s)
            s=re.sub(r'\)', '', s)
            dataset_x.append(s)
            dataset_y.append(1)
            i+=1
    test_data = []
    test_ids = []
    with open('test.txt',encoding="utf8") as f:
        for data in f.readlines():    
            s = (data.split('\t',1)[1].lower())
            s=re.sub('\d', '', s)
            s=re.sub(r'\(', '', s)
            s=re.sub(r'\)', '', s)
            test_data.append(s)
            test_ids.append(data.split('\t',1)[0])
            i+=1
    print(len(dataset_x),len(dataset_y))
    # Split dataset
    X_train, y_train = dataset_x, dataset_y
    X_test = test_data

    feat = Featurizer()

    
   
    
    feat_train = feat.train_feature(X_train)
    feat_test = feat.test_feature(X_test)

    lr = SGDClassifier(max_iter=10,loss='log',alpha=0.000001,penalty='l2',learning_rate='optimal',verbose=2,shuffle=True)
    alphalist = [0.0001,0.000001,0.00001]
    scores=[]
    for alpha in alphalist:
        lr.alpha = alpha
        this_scores = cross_val_score(lr, feat_train, y_train, n_jobs=1)
        scores.append(np.mean(this_scores))
        print("Accuracy on training set with alpha=",alpha," is :", np.mean(this_scores))
    bestalpha = alphalist[scores.index(max(scores))]
    lr.alpha =bestalpha
    print('------------------------------------')
    print("Best alpha parameter:", bestalpha)
    print('------------------------------------')
    
    lr.fit(feat_train, y_train)
    y_pred = lr.predict(feat_test)
    output = open("C://Users/bhavana/Desktop/fall-2017/Natural Language Processing -CSCI 5832-001/assign-3/output3.txt",'w')
    for j in range(len(y_pred)):
        if y_pred[j] == 1:
            print(test_ids[j],"POS")
            output.write(test_ids[j]+"\t"+"POS\n")
        else :
            print(test_ids[j],"NEG")
            output.write(test_ids[j]+"\t"+"NEG\n")
    