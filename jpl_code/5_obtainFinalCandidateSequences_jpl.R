library(tidyverse)

#File Pathnames. Change them to match yours.
input <- "/Users/joseparedes/Desktop/kappelLab/structuredDomainLibrary/4_domainLibraryPhysicalProperties.tsv"
output <- "/Users/joseparedes/Desktop/kappelLab/structuredDomainLibrary/5_finalCandidateSequences.tsv"
df <- read_tsv(input, na=c("", "NA"))

#Ensure columns holding metrics are numeric
df <- df %>% mutate(
    interactionIndex=as.numeric(interactionIndex),
    aromaticSurfaceFraction=as.numeric(aromaticSurfaceFraction),
    `Rg(Compactness)`=as.numeric(`Rg(Compactness)`),
    fractionBuried=as.numeric(fractionBuried),
    positiveSurfaceFraction=as.numeric(positiveSurfaceFraction),
    negativeSurfaceFraction=as.numeric(negativeSurfaceFraction))

#Compute additional charge metrics
df <- df %>% mutate(
    netSurfaceCharge=positiveSurfaceFraction - negativeSurfaceFraction,
    totalChargeFraction=positiveSurfaceFraction + negativeSurfaceFraction)

#Store rows containing all required values into a dataframe for plotting
dfPlot <- df %>% filter(
    !is.na(interactionIndex),
    !is.na(aromaticSurfaceFraction),
    !is.na(`Rg(Compactness)`),
    !is.na(totalChargeFraction),
    !is.na(netSurfaceCharge))

#Define percentile-based thresholds
lowInteractionThreshold <- quantile(dfPlot$interactionIndex, 0.30) #Interaction index threshold below which 30% of domains fall
highAromaticThreshold <- quantile(dfPlot$aromaticSurfaceFraction, 0.75) #Domains above this aromatic surface fraction are in the top 25%
highChargeThreshold <- quantile(dfPlot$totalChargeFraction, 0.75) #Domains above this charge surface fraction are in the top 25%

rgLower <- quantile(dfPlot$`Rg(Compactness)`, 0.25) #We want domains that are in the middle 50% of compactness
rgUpper <- quantile(dfPlot$`Rg(Compactness)`, 0.75)

#Visualize how many sequences influence condensation through aromatic interactions
aromaticPlot <- ggplot(dfPlot, aes(x=interactionIndex, y=aromaticSurfaceFraction, size=`Rg(Compactness)`, color=fractionBuried))+
  geom_point(alpha=0.8)+
  annotate("rect",
           xmin=-Inf,
           xmax=lowInteractionThreshold,
           ymin=highAromaticThreshold,
           ymax=Inf,
           alpha=0.1,
           fill="red")+
  scale_color_viridis_c(option="plasma")+
  labs(
    x="Interaction Index with Parent Protein",
    y="Aromatic Surface Fraction",
    color="Fraction Buried",
    size="Rg (Å)")+
  theme_minimal(base_size=14)
print(aromaticPlot)

#Visualize how many sequences influence condensation through charge interactions
chargePlot <- ggplot(dfPlot, aes(x=interactionIndex, y=totalChargeFraction, size=`Rg(Compactness)`, color=netSurfaceCharge))+
  geom_point(alpha=0.8)+
  annotate("rect",
           xmin=-Inf,
           xmax=lowInteractionThreshold,
           ymin=highChargeThreshold,
           ymax=Inf,
           alpha=0.1,
           fill="blue")+
  scale_color_gradient2(low="blue", mid="white", high="red")+
  labs(
    x="Interaction Index",
    y="Total Charged Surface Fraction",
    color="Net Surface Charge",
    size="Rg (Å)")+
  theme_minimal(base_size=14)
print(chargePlot)

#Determine if sequences influence condensate formation through aromatic interactions, charge interactions, both, or neither
dfPlot <- dfPlot %>% mutate(
    aromaticCandidate =
      interactionIndex < lowInteractionThreshold &
      aromaticSurfaceFraction > highAromaticThreshold &
      `Rg(Compactness)` > rgLower &
      `Rg(Compactness)` < rgUpper,
    
    chargeCandidate =
      interactionIndex < lowInteractionThreshold &
      totalChargeFraction > highChargeThreshold &
      `Rg(Compactness)` > rgLower &
      `Rg(Compactness)` < rgUpper,
    
    candidateSequence = case_when(
      aromaticCandidate & chargeCandidate ~ "Both",
      aromaticCandidate ~ "Aromatic-driven",
      chargeCandidate ~ "Charge-driven",
      TRUE ~ "Neither"))

#Visualize the final candidate sequences
candidatesPlot <- ggplot(dfPlot, aes(x=interactionIndex, y=aromaticSurfaceFraction, color=candidateSequence, size=`Rg(Compactness)`))+
  geom_point(alpha=0.8) +
  scale_color_manual(values=c(
    "Aromatic-driven"="purple",
    "Charge-driven"="blue",
    "Both"="red",
    "Neither"="grey"))+
  labs(
    x="Interaction Index",
    y="Aromatic Surface Fraction",
    size="Rg (Å)",
    color="Driver Class")+
  theme_minimal(base_size=14)
print(candidatesPlot)

#Save the final candidate sequences
finalCandidateSequences <- dfPlot %>%
  select(Entry, Domain, `Domain Sequence`, Start, End, candidateSequence) %>%
  arrange(candidateSequence)
write_tsv(finalCandidateSequences, output)




