import json
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import time
import gzip
import os
import torch
from blingfire import text_to_sentences
from tqdm import tqdm


if not torch.cuda.is_available():
  print("Warning: No GPU found. Please add GPU to your notebook")


# We use the Bi-Encoder to encode all passages, so that we can use it with sematic search
model_name = 'nq-distilbert-base-v1'
bi_encoder = SentenceTransformer(model_name)
#top_k = 5 


def get_top_k_sentences(passage,query,top_k):
    corpus_embeddings = bi_encoder.encode(passage, convert_to_tensor=True, show_progress_bar=True)
    question_embedding = bi_encoder.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=top_k)
    hits = hits[0]
    #print(hits)
    relevant_sents =[]
    for hit in hits:
        #print("\t{:.3f}\t{}".format(hit['score'], passage[hit['corpus_id']]))
        relevant_sents.append(passage[hit['corpus_id']])
    return relevant_sents

    






if __name__ == "__main__":
    #passage = "Emmitt James Smith III ( born May 15 , 1969 ) is an American former professional football player who was a running back for fifteen seasons in the National Football League ( NFL ) during the 1990s and 2000s , primarily with the Dallas Cowboys . A three-time Super Bowl champion with the Cowboys , he is the league 's all-time leading rusher . Smith grew up in Pensacola , Florida and became the second-leading rusher in American high school football history while playing for Escambia High School . Smith played three years of college football for the Florida Gators , where he set numerous school rushing records . After being named a unanimous All-American in 1989 , Smith chose to forgo his senior year of eligibility and play professionally . The Cowboys selected Smith in the first round of the 1990 NFL draft . During his long professional career , he rushed for 18,355 yards , breaking the record formerly held by Walter Payton . He also holds the record for career rushing touchdowns with 164 . Smith is the only running back to ever win a Super Bowl championship , the NFL Most Valuable Player award , the NFL rushing crown , and the Super Bowl Most Valuable Player award all in the same season ( 1993 ) . He is also one of four running backs to lead the NFL in rushing three or more consecutive seasons , joining Steve Van Buren , Jim Brown and Earl Campbell . Smith led the league in rushing and won the Super Bowl in the same year three times ( 1992 , 1993 , and 1995 ) when to that point it had never been done . Smith is also one of only two non-kickers in NFL history to score more than 1,000 career points ( the other being Jerry Rice ) .The Dallas Cowboys are a professional American football team based in the Dallas-Fort Worth metroplex . The Cowboys compete in the National Football League ( NFL ) as a member club of the league 's National Football Conference ( NFC ) East division . The team is headquartered in Frisco , Texas , and plays its home games at AT & T Stadium in Arlington , Texas , which opened for the 2009 season . The stadium took its current name prior to the 2013 season . The Cowboys joined the NFL as an expansion team in 1960 . The team 's national following might best be represented by its NFL record of consecutive sell-outs . The Cowboys ' streak of 190 consecutive sold-out regular and post-season games ( home and away ) began in 2002 . The franchise has made it to the Super Bowl eight times , tied with the Pittsburgh Steelers and the Denver Broncos for second most Super Bowl appearances in history , just behind the New England Patriots record eleven Super Bowl appearances . This has also corresponded to eight NFC championships , most in the NFC . The Cowboys have won five of those Super Bowl appearances , tying them with their NFC rivals , the San Francisco 49ers ; both are second to Pittsburgh 's and New England 's record six Super Bowl championships . The Cowboys are the only NFL team to record 20 straight winning seasons ( 1966-85 ) , in which they missed the playoffs only twice ( 1974 and 1984 ) . In 2015 , the Dallas Cowboys became the first sports team to be valued at $ 4 billion , making it the most valuable sports team in the world , according to Forbes . The 1990 NFL season was the 71st regular season of the National Football League . To increase revenue , the league , for the first time since 1966 , reinstated bye weeks , so that all NFL teams would play their 16-game schedule over a 17-week period . Furthermore , the playoff format was expanded from 10 teams to 12 teams by adding another wild card from each conference , thus adding two more contests to the postseason schedule ; this format remains in use today ( although there are now four division spots and two wild card spots available with realignment in 2002 ) . During four out of the five previous seasons , at least one team with a 10-6 record missed the playoffs , including the 11-5 Denver Broncos in 1985 ; meanwhile , the 10-6 San Francisco 49ers won Super Bowl XXIII , leading for calls to expand the playoff format to ensure that 10-6 teams could compete for a Super Bowl win . Ironically , the first ever sixth-seeded playoff team would not have a 10-6 record , but instead , the New Orleans Saints , with an 8-8 record , took the new playoff spot . This was also the first full season for Paul Tagliabue as the league 's Commissioner , after taking over from Pete Rozelle midway through the previous season . ABC was given the rights to televise the two additional playoff games . Meanwhile , Turner 's TNT network started to broadcast Sunday night games for the first half of the season . On October 8 , the league announced that the Super Bowl Most Valuable Player Award would be named the Pete Rozelle Trophy . The season ended with Super Bowl XXV when the New York Giants defeated the Buffalo Bills 20-19 at Tampa Stadium . This would be the first Super Bowl appearance for Buffalo , who would lose the next three Super Bowls as well . Late in the season , with the Gulf War looming closer , the NFL announced that starting in Week 16 ( and continuing until Super Bowl XXV ) , the league would add American flag decals to the back of the helmet . The 2002 NFL season was the 83rd regular season of the National Football League . The league went back to an even number of teams , expanding to 32 teams with the addition of the Houston Texans ; the league has since remained static with 32 teams since . The clubs were then realigned into eight divisions , four teams in each . Also , the Chicago Bears played their home games in 2002 in Champaign , Illinois at Memorial Stadium because of the reconstruction of Soldier Field . The NFL title was eventually won by the Tampa Bay Buccaneers when they defeated the Oakland Raiders 48-21 in Super Bowl XXXVII , at Qualcomm Stadium in San Diego , California on January 26 , 2003 . The Arizona Cardinals are a professional American football franchise based in Phoenix , Arizona . They compete in the National Football League ( NFL ) as a member club of the National Football Conference ( NFC ) West division . The Cardinals were founded as the Morgan Athletic Club in 1898 , and are the oldest continuously run professional football team in the United States . The Cardinals play their home games at State Farm Stadium , which opened in 2006 and is located in the northwestern suburb of Glendale . The team was established in Chicago in 1898 as an amateur football team and joined the NFL as a charter member on September 17 , 1920 . Along with the Chicago Bears , the club is one of two NFL charter member franchises still in operation since the league 's founding ( the Green Bay Packers were an independent team until they joined the NFL a year after its creation in 1921 ) . The club then moved to St. Louis in 1960 and played in that city through 1987 ( sometimes referred to as the Football Cardinals or the Big Red to avoid confusion with the St. Louis Cardinals of Major League Baseball ) . Before the 1988 season , the team moved west to Tempe , Arizona , a college suburb east of Phoenix , and played their home games for the next 18 seasons at Sun Devil Stadium on the campus of Arizona State University . In 2006 , the club moved to their current home field in Glendale , although the team 's executive offices and training facility remain in Tempe . The franchise has won two NFL championships , both while it was based in Chicago . The first occurred in 1925 , but is the subject of controversy , with supporters of the Pottsville Maroons believing that Pottsville should have won the title . Their second title , and the first to be won in a championship game , came in 1947 , nearly two decades before the first Super Bowl . The 2003 NFL season was the 84th regular season of the National Football League ( NFL ) . Regular-season play was held from September 4 , 2003 , to December 28 , 2003 . Due to damage caused by the Cedar Fire , Qualcomm Stadium was used as an emergency shelter , and thus the Miami Dolphins-San Diego Chargers regular-season match on October 27 was instead played at Sun Devil Stadium , the home field of the Arizona Cardinals . The playoffs began on January 3 , 2004 . The NFL title was won by the New England Patriots when they defeated the Carolina Panthers , 32-29 , in Super Bowl XXXVIII at Reliant Stadium in Houston , Texas , on February 1 . This was the last season until the 2016 NFL season where neither of the previous Super Bowl participants made the playoffs . The 2004 NFL season was the 85th regular season of the National Football League . With the New England Patriots as the defending league champions , regular season play was held from September 9 , 2004 to January 2 , 2005 . Hurricanes forced the rescheduling of two Miami Dolphins home games : the game against the Tennessee Titans was moved up one day to Saturday , September 11 to avoid oncoming Hurricane Ivan , while the game versus the Pittsburgh Steelers on Sunday , September 26 was moved back 7\u00bd hours to miss the eye of Hurricane Jeanne . The playoffs began on January 8 , and eventually New England repeated as NFL champions when they defeated the Philadelphia Eagles 24-21 in Super Bowl XXXIX , the Super Bowl championship game , at ALLTEL Stadium in Jacksonville , Florida on February 6 ."
    #query = "Jerry what is the middle name of the player with the second most National Football League career rushing yards ?"
    #passages = text_to_sentences(passage).split("\n")

    data = json.load(open("data/processed_data/test_processed_new.json"))
    new_data =[]
    for d in tqdm(data):
        qid = d['question_id']
        passage = d['table_passage_row']
        passages = text_to_sentences(passage).split("\n")
        query = d['question']
        #print(query)
        #print(d['answer-text'])
        sents_relevant_to_the_ques = top_3_sentences = get_top_k_sentences(passages,query,3)
        sents_relevant_to_the_ques = " ".join(sents_relevant_to_the_ques)
        #print(sents_relevant_to_the_ques)
        d['table_passage_row'] = sents_relevant_to_the_ques
        new_data.append(d)
    json.dump(new_data,open("data/processed_data/test_processed_new_with_rel_sents.json","w"))