"""
R script necessary to analyse the thickness of different tissus
"""


library(tidyverse)
library(magrittr)
library(agricolae)
library(cowplot)

######## theme  #######
theme_set(theme_cowplot(font_family = 'Times',font_size = 15,
                        rel_small = .95,line_size = 1,rel_tiny = 1.5,rel_large = 1.5
))

theme_set(theme_cowplot(line_size = 0.7,font_family = 'Times',
                        rel_small = 12/14, rel_tiny = 11/14, rel_large = 16/14))

theme_set(theme_cowplot())

theme_set(theme_cowplot(line_size = 0.7,font_family = 'Helvetica',
                        rel_small = 12/14, rel_tiny = 11/14, rel_large = 16/14))
theme_set(theme_bw(base_size = 14,base_rect_size = 1.5,base_line_size = 0.5,base_family = 'Helvetica'))

bold.16.text <- element_text(face = "bold", size = 16,colour = 'black')
bold.20.text <- element_text(face = "bold", size = 20,colour = 'black')
text.16 <- element_text( size = 16,colour = 'black')
text.14 <- element_text( size = 14,colour = 'black')

####Data#####

tab <-  read.csv('../tableofvaluesforRstudio.csv')
#becarefull, in this table X column corresponds to the index table in python
#attention dans le tab, il y a l'index du tab sous python
tab %>% head()


#######raw representation ###########
#Boxplot toutes les conditions les unes a cote des autres 
#boxplot of all conditions next to each other
tab %>% 
  gather(key = 'TRT',value = 'Distance_pixel',convert = T,-X) %>% 
  ggplot(aes(x = TRT, y = Distance_pixel))+
  geom_boxplot()

#tableau de moyennes et sd tous les points en meme temps
#table of means and sd of all points in the same time
tab %>% 
  gather(key = 'TRT',value = 'Distance_pixel',convert = T,-X,na.rm = T) %>% #head
  separate(TRT, into =c('truc_1','truc_2','treatment','repet','code'),sep = '_', remove = F) %>%
  group_by(truc_2,treatment) %>% 
  summarize(m = mean(Distance_pixel),
            sd = sd(Distance_pixel)) %>% 
  view()
#violon plot truc_2 facet_grid
tab %>% 
  gather(key = 'TRT',value = 'Distance_pixel',convert = T,-X,na.rm = T) %>% #head
  separate(TRT, into =c('truc_1','truc_2','treatment','repet','code'),sep = '_', remove = F) %>% 
  ggplot(aes(x = treatment, y = Distance_pixel))+
  geom_violin()+
  facet_grid(truc_2~.)


#violon plot truc_2 cote a cote
tab %>% 
  gather(key = 'TRT',value = 'Distance_pixel',convert = T,-X,na.rm = T) %>% #head
  separate(TRT, into =c('truc_1','truc_2','treatment','repet'),sep = '_', remove = F) %>% 
  ggplot(aes(x = treatment, y = Distance_pixel, fill = truc_2))+
  geom_violin()
#boxplotplot truc_2 cote a cote
tab %>% 
    gather(key = 'TRT',value = 'Distance_pixel',convert = T,-X,na.rm = T) %>% #head
    separate(TRT, into =c('truc_1','truc_2','treatment','repet'),sep = '_', remove = F) %>% 
    ggplot(aes(x = treatment, y = Distance_pixel, fill = truc_2))+
   
    geom_boxplot()
  
# boxplot, mean per repeat
tab %>% 
  gather(key = 'TRT',value = 'Distance_pixel',convert = T,-X,na.rm = T) %>% #head
  separate(TRT, into =c('truc_1','truc_2','treatment','repet','code'),sep = '_', remove = F) %>% 
  group_by(truc_2,treatment,repet) %>% 
  summarize(m = mean(Distance_pixel),
           sd = sd(Distance_pixel)) %>% 
  ggplot(aes(x = treatment, y = m, fill = truc_2))+
  geom_boxplot(outlier.colour = 'red')+
  geom_jitter(width = 0.4)+
  facet_grid(.~truc_2)
  





####### exlusion of outlieres######
#boxplot en facet truc_2, mets cote a cote les expe
tab %>% 
  gather(key = 'TRT',value = 'Distance_pixel',convert = T,-X,na.rm = T) %>% #head
  separate(TRT, into =c('truc_1','truc_2','treatment','repet','code'),sep = '_', remove = F) %>% 
  mutate(code2 = str_c(truc_1,truc_2,treatment,repet,sep = '_')) %>% 
  ggplot(aes(x = code2, y = Distance_pixel,fill = treatment))+
  geom_boxplot()+
  facet_wrap(truc_2~., scales = 'free_x')

#box plot, facet truc_2, regroupe per treatment
tab %>% 
  gather(key = 'TRT',value = 'Distance_pixel',convert = T,-X,na.rm = T) %>% #head
  separate(TRT, into =c('truc_1','truc_2','treatment','repet','code'),sep = '_', remove = F) %>% 
  mutate(code2 = str_c(truc_1,truc_2,treatment,repet,sep = '_')) %>% 
  
  ggplot(aes(x = treatment, y = Distance_pixel,fill = treatment))+
  geom_boxplot()+
  facet_wrap(truc_2~., scales = 'free_x')



##modification of the table. One picture will be composed by the left and the right site
table<-  tab %>% 
  gather(key = 'TRT',value = 'Distance_pixel',convert = T,-X,na.rm = T) %>% #head
  separate(TRT, into =c('truc_1','truc_2','treatment','repet','code'),sep = '_', remove = F) %>% 
  mutate(code2 = str_c(truc_1,truc_2,treatment,repet,sep = '_')) 
table %>% head
code2_unique <- unique(table$code2)


#fonction which exlude outliers : Iteration 5 times
filter_lst_outliers  <- function(lst){
  
  o = boxplot.stats(lst)$out
  t1 =  lst[!lst %in% o]
  o1 = boxplot.stats(t1)$out
  t2 = t1[!t1 %in% o1]
  o2 = boxplot.stats(t2)$out
  t3 = t2[!t2 %in% o2]
  o3 = boxplot.stats(t3)$out
  t4 = t3[!t3 %in% o3]
  o4 = boxplot.stats(t4)$out
  t5 = t4[!t4 %in% o4]
  
  return(t5)
}
# For loop that performe the filtering
nouvelle_lst <- c()
for(i in 1:length(code2_unique)){
  txt = code2_unique[i]
  tab1 = table %>% filter(code2 == txt)
  tab1 %<>% filter (Distance_pixel %in% filter_lst_outliers(tab1$Distance_pixel))
  nouvelle_lst[i] = list(tab1)
}
#table with  filtered values
tab_filtered <- do.call(rbind, nouvelle_lst)

tab_filtered %>% head()

#boxplot facet truc-2, treatment cote a cote
tab_filtered %>% 
  ggplot(aes(x = code2, y = Distance_pixel,fill = treatment))+
  geom_boxplot()+
  facet_wrap(truc_2~., scales = 'free_x')


tab_filtered %>% #head()
  #gather(key = 'TRT',value = 'Distance_pixel',convert = T,-X,na.rm = T) %>% #head
  #separate(TRT, into =c('truc_1','truc_2','treatment','repet','code',),sep = '_', remove = F) %>%
  group_by(truc_2,treatment) %>% 
  summarize(m = mean(Distance_pixel),
            sd = sd(Distance_pixel),
            m_mi = m/2.1502,
            sd_mi = sd/2.1502) %>% 
  view()

tab_filtered %>% mutate(Distance_micro = Distance_pixel/2.1502) %>% 
  ggplot(aes(x = treatment, y = Distance_micro,fill = treatment))+
  geom_boxplot()+
  facet_wrap(truc_2~., scales = 'free_x')

tab_filtered %>%
  mutate(Distance_micro = Distance_pixel/2.1502) %>% 
  group_by(truc_2,treatment,repet) %>% 
  summarize(m = mean(Distance_pixel),
            sd = sd(Distance_pixel),
            m_mi = mean(Distance_micro),
            sd_mi = sd(Distance_micro)) %>%
  ggplot(aes(x = treatment, y = m_mi, fill = truc_2))+
  geom_boxplot(outlier.colour = 'red')+
  geom_jitter(width = 0.4)+
  theme(legend.position="none",
        axis.text.x = element_text(face = "bold", color = "#993333", 
                                                        #size = 12, 
                                                        angle = 45,
                                                        hjust = 1))+
  facet_grid(.~truc_2)+
  ylim(0,35)+
  ylab('Thickness µm')+
  xlab('Condition')

#plot qui reprends tous les points sélectionnés par conditions et par replicats
tab_filtered %>%
  mutate(Distance_micro = Distance_pixel/2.1502) %>% #head
  ggplot(aes(x = repet, y = Distance_micro))+
  geom_boxplot()+
  geom_jitter(width = 0.2, size =0.1)+
  facet_wrap(truc_2~ treatment,scales = 'free_x' )+
  theme(axis.text.x = element_text(face = "bold", color = "#993333", 
                                   #size = 12, 
                                   angle = 45,
                                   hjust = 1))+
  ylab('Thickness µm')+
  xlab('replicate')+
  ylim(0,70)
#1600-800       
#####tableau#####
tab_filtered %>%mutate(Distance_micro = Distance_pixel/2.1502) %>% head#
write_csv('../Complete_table_filtered.csv')

tab_filtered %>%
  mutate(Distance_micro = Distance_pixel/2.1502) %>% 
  group_by(truc_2,treatment,repet) %>% 
  summarize(m = mean(Distance_pixel),
            sd = sd(Distance_pixel),
            m_mi = mean(Distance_micro),
            sd_mi = sd(Distance_micro)) %>% head#view#head
  write_csv('../Table_mean_per_pictures.csv')

tab_filtered %>%
  mutate(Distance_micro = Distance_pixel/2.1502) %>% 
  group_by(truc_2,treatment,repet) %>% 
  summarize(m = mean(Distance_pixel),
            sd = sd(Distance_pixel),
            m_mi = mean(Distance_micro),
            sd_mi = sd(Distance_micro)) %>%
  group_by(truc_2,treatment) %>% 
  summarize(m = mean(m_mi),
            sd = sd(m_mi)) %>% head()
  write_csv('../Table_mean_per_conditions.csv')
##### statistics #####
#model mix
tab_for_stats <- tab_filtered %>%
    mutate(Distance_micro = Distance_pixel/2.1502,
           code3 = str_c(truc_2,treatment, sep= '__'))# %>% head#



library(lme4)
#variable aleatoire est le point de lecture
lmer1 <- lmer(Distance_micro~code3+(1|code2),  data=tab_for_stats)
summary(lmer1)

library(multcomp)
summary(glht(lmer1,linfct = mcp(code3 = "Tukey")))
pc1 <- glht(lmer1,linfct = mcp(code3 = "Tukey"))

cld(pc1, level=0.001)

#model lineaire classique par photo
tab_for_stats2<- tab_filtered %>%
  mutate(Distance_micro = Distance_pixel/2.1502) %>% 
  group_by(truc_2,treatment,repet) %>% 
  dplyr::summarize(m = mean(Distance_pixel),
            sd = sd(Distance_pixel),
            m_mi = mean(Distance_micro),
            sd_mi = sd(Distance_micro)) %>% 
  mutate(code3 = str_c(truc_2,treatment,sep = '__'))
library(agricolae)
aov <-  aov(m_mi~code3,  data=tab_for_stats2)
summary(aov)

lm2 <- lm(m_mi~code3,  data=tab_for_stats2)
summary(lm2)
HSD.test(lm2 , trt='code3',console = T)
HSD.test(aov , trt='code3',console = T)

