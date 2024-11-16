from typing import List, Tuple, Union, Any, Dict
import networkx as nx
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
import torch

from .functions import encode, select_top_k, generate_text

class TextRank():
    def __init__(self, model_name:str, distance_metric:str) -> None:
        self.model = SentenceTransformer(model_name)
        self.metric = distance_metric
        nltk.download('punkt')
    
    def extract_text(self, text: str, sbert: SentenceTransformer,  top_k: int) -> str:
        sentences, embeddings = encode(text, sbert)
        ranked_sentences = self.textrank_raw(sentences, embeddings)
        ranked_sentences = select_top_k(ranked_sentences, top_k)
        return generate_text(ranked_sentences)

    def encode(self, text:str) -> Tuple[List[str], torch.Tensor]:
        return encode(text, self.model)
    
    def textrank(self, sentences: List[str], embeddings: torch.Tensor, top_k: int, return_text: bool = False) -> Union[str, List[Tuple[float, str]]]:
        ranked_sentences = self.textrank_raw(sentences, embeddings)
        ranked_sentences = select_top_k(ranked_sentences, top_k)
        if return_text:
            return generate_text(ranked_sentences)
        else:
            return ranked_sentences

    def textrank_raw(self, sentences: List[str], embeddings: torch.Tensor) -> Dict[str, List[Union[float, int, str]]]:
        embeddings = embeddings.cpu()
        if self.metric == "cosine_similarity":
            reverse = True
            mat = cosine_similarity(embeddings, embeddings)
        elif self.metric == "euclidean_distance":
            reverse = False
            mat = euclidean_distances(embeddings, embeddings)
        elif self.metric == 'dot_product':
            reverse = True
            mat = util.dot_score(embeddings, embeddings).numpy()

        graph = nx.from_numpy_matrix(mat)
        pr = nx.pagerank(graph, max_iter=20000)
        ranked_sentences = sorted(((pr[i], i,s) for i,s in enumerate(sentences)), reverse=reverse)
        ranked_sentences = list(zip(*ranked_sentences))
        ranked_sentences = {
            'ranks': list(ranked_sentences[0]),
            'positions': list(ranked_sentences[1]),
            'sentences': list(ranked_sentences[2])
        }
        return ranked_sentences

if __name__ == "__main__":
    def main():
        ranker = TextRank("all-MiniLM-L6-v2", "dot_product")
        sentences, embeddings = ranker.encode(text)
        sentences = ranker.textrank(sentences, embeddings, 5, True)
        print(sentences)
        for i in range(5):
            print(sentences[i][1], '\n')


    text = """
    Christmas week brings an exciting labour market win for Rishi Sunak, as the prime minister finally manages to appoint an ethics adviser. For the past few months, this empty role has begun to look like one of the jobs that British people no longer seem minded to do, like fruit picking or being Nigel Farage’s wife. As part of Sunak’s commitment to being the change he wants to see, his ethics guy is Sir Laurie Magnus, a former investment banker who won’t have the power to launch his own investigations. Maybe this is Sunak’s version of a terrible cracker joke.

A full 18 hours into his job, Magnus has yet to stage a wildcat strike, which feels like a rare industrial relations success for the prime minister. For those struggling to keep track of who else is on strike, a useful rule of thumb is that if a secretary of state has dressed up as one on a visit this year, they are now taking industrial action.

Senior politicians have cosplayed as train drivers, ambulance workers, Border Force officials – the list goes on. We’ve yet to see health secretary Steve Barclay in a nurse’s uniform, though do assume that the reason politicians tuck their tie into their shirt when they visit hospitals is to prevent someone grabbing them by it and asking them what on earth they actually meant when they ostentatiously clapped for carers. It says a lot that you can be booked for sarcastic applause in football, but in politics it can see you promoted.

As for how it’s all going down with the public, the Chinese this week unveiled an ultra-deepwater drillship that will be able to plumb twice the depth at which the Titanic rests, though that still leaves the Conservative poll rating just beneath its reach. With the country settling into a kind of perma-rage that nothing much works any more, there is something mesmeric about the government’s attempts to insist its dignity hasn’t been compromised, and that it has taken back control of the taking back control. The party’s chaos machine has spewed out just the three prime ministers this year, yet Sunak’s appearance before the liaison committee this week appeared to downplay this farce to the equivalent of a few substitutions in your shopping order. Sorry, Downing Street contains the following substitutions: 1xLizTruss for 1xBorisJohnson, 1xRishiSunak for 1xLizTruss, 1xEvenWorseEconomicProspects for 1xBadEconomic Prospects.

Yet for all the dysfunction and breakdown taking place out there in the place we call reality, Sunak comes across as a sort of prime ministerial chat tool, a state-of-the-art robot whose learned responses are uncannily human-adjacent, but divorced from any sense he meaningfully gets any of it. “I’m really, really robust,” he told the liaison committee, which feels like the sort of thing Alexa’s software throws up after a slight pause when your kids ask it a rude question.

'We want to save lives': ambulance staff strike across England and Wales – video report
01:56
'We want to save lives': ambulance staff strike across England and Wales – video report
Then again, perhaps this is where he wishes to take the role. Sunak has long been one of those Silicon Valley-frotters who has swallowed all the bullshit about the possibility of a frictionless world, where tech companies act in concert to provide superior services to the muddle of the state, and politicians like him are willingly reduced to a kind of genial front-of-house role. They are not so much problem-solvers and pathfinders as polished maître-d’ figures, selected only because people still like to see a human in a ceremonial front-facing role, even though they know the country is essentially driverless.

It does feel like we’re half the way there. The country certainly appears driverless – but mainly in the sense that Brian Harvey’s car was when the East 17 singer contrived to run himself over with it.

Of course, the reality of all the tech bros’ driverless utopias is that beneath the supposedly unruffled surface are the countless exhausted workers it takes to keep the appearance of seamlessness on the road. A significant number of these are joining the strikes, somehow unpersuaded that their reclassification as “key workers” during the pandemic came from the heart. An early readout of Covid was that middle-class people stayed at home while working-class people brought them stuff. Or saved their lives, or whatever. The current attempt to rebrand key workers as uppity workers is certainly bold. Unfortunately, it’s proving quite difficult to shift the perception that all emissaries of the government actively make everything worse. A plan to include ministers among volunteers manning Border Force positions during the walkout has failed to persuade even No 10. A senior government official conceded to Friday’s Times: “Having a minister and their entourage is probably the last thing that people need to minimise the disruption.”

And that perception feels public-sector-wide. As the UK’s post-pandemic recovery is revealed to have been even weaker than previously thought, it’s almost as if ministers dressing up in the clothes of people who do essential jobs has reached its limits as a pageantry strategy. The UK is the only G7 country not to have regained the ground lost during the lockdown, with seemingly no politician willing to face up to an answer or range of answers as to why that may be. The government comes across as something events happen to, rather than because of, except in a negative way. Still, nurses have this morning announced further strikes next month, so perhaps some of them could dress up as politicians for a day and go on a site visit to Westminster to man the frontline themselves. It’s increasingly difficult to see how they could do any worse.
    """
    main()