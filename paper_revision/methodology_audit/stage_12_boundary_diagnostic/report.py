"""Report edge enrichment alongside the original ring-controlled spatial test."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent


def main():
    t=pd.read_csv(OUT/'tail_boundary_enrichment.csv');e=pd.read_csv(OUT/'edge_effects.csv');i=pd.read_csv(OUT/'interior_sensitivity.csv');r=pd.read_csv(OUT/'ring_diagnostic.csv');profiles=pd.read_csv(OUT/'ring_profiles.csv')
    samples=['NCBI785','NCBI784','NCBI783'];primary=t[t.distance.eq('tissue_distance_um')&t.band_um.eq(200)&t['tail'].eq(.1)&t.group.eq('upper')].copy()
    primary['edge_fraction']=primary.n_edge/primary.n_total
    # Patient/group summary of absolute tail-rate differences avoids infinite risk ratios.
    t['rate_difference']=t.edge_tail_rate-t.interior_tail_rate
    keys=['family','unit','distance','band_um','tail','group'];unit=t.groupby(keys)[['edge_tail_rate','interior_tail_rate','rate_difference']].mean().reset_index();unit.to_csv(OUT/'unit_tail_boundary_effects.csv',index=False)
    overview=t.groupby(['family','distance','band_um','tail','group']).agg(n_sections=('relative_risk','size'),n_finite_ratios=('relative_risk','count'),n_edge_enriched=('relative_risk',lambda x:int((x>1).sum())),median_relative_risk=('relative_risk','median')).reset_index();overview.to_csv(OUT/'boundary_summary.csv',index=False)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(1,3,figsize=(11,3.5),layout='constrained')
    for axis,sample in zip(ax,samples):
        a=profiles[profiles['sample'].eq(sample)]
        axis.plot(a.ring+1,a.upper_decile_rate,'o-',color='#bd3935',label='Upper decile');axis.plot(a.ring+1,a.lower_decile_rate,'o-',color='#2879ad',label='Lower decile');axis.axhline(.1,color='.6',ls='--');axis.set(xlabel='Hull-distance band: edge → interior',ylabel='Fraction of spots in tail',title=sample,xticks=[1,5,10],ylim=(0,max(.3,a[['upper_decile_rate','lower_decile_rate']].max().max()+.03)))
    ax[0].legend(fontsize=8);fig.savefig(OUT/'ring_tail_profiles.png',dpi=180);plt.close(fig)
    md=lambda a:a.to_markdown(index=False,floatfmt='.3f')
    lines=['# Boundary effects and the existing ring control','',
      '**Yes: the original analysis includes a relevant boundary-ring permutation control. The maps also show measurable boundary enrichment. Those findings are compatible.** This exploratory follow-up was prompted by the author’s inspection after the decile maps, with widths/geometries frozen before the new comparisons. It leaves predictors, scores and primary groups unchanged.','',
      '## What the original code does','',
      '`assign_boundary_rings()` measures distance to the convex hull of analyzed spot centers, divides those distances into ten quantile bands, and merges undersized groups. These are edge-distance bands, not literal circles centered on the image. Script 06 shuffles the unchanged conditional scores only within those bands when constructing the Moran permutation reference. That preserves every band’s score distribution and hence a broad bandwise edge gradient. It asks whether arrangement within/between those fixed bands gives more spatial autocorrelation than that restricted shuffling.','',
      'It does not subtract an edge trend from individual scores or change Q1/Q4. The corrected archived IDC ring-controlled p-values are .001 in all 18 sections (999 permutations, minimum attainable .001). Their observed Moran values reproduce against the fixed scores. These are conditional spatial tests with within-band exchangeability assumptions, not proof of independence from tissue geometry or of biological cause. The convex hull ignores holes/concavities and does not distinguish tissue, image and expression-coverage boundaries.','',
      'The earlier “interior-only DE” script is a different control: it chooses locations by similarity of neighboring histology embeddings and then re-ranks the retained scores. It did not erode physical boundary bands. Neither this distinction nor visible edge enrichment negates the useful original ring test.','',
      '## Direct boundary measurements','',
      'For the upper 10%, comparison within 200 µm of the actual saved H&E tissue-mask boundary versus farther inside:','',md(primary[primary['sample'].isin(samples)][['sample','n_edge','edge_fraction','edge_tail_rate','interior_tail_rate','relative_risk','tail_fraction_in_edge']]),'',
      'Rates are fractions, not percentages; relative risk is near-boundary rate divided by interior rate. The high tail is enriched near this boundary in 26/32 sections (8/11 discovery, 7/7 validation, 4/4 COAD, 7/10 Visium). These are descriptive section counts, not independent-patient discoveries. All sections, lower tails, 25% tails and both widths are in the saved tables. [Patient/known-group rate differences](unit_tail_boundary_effects.csv) average repeated sections within unit.','',
      'Different boundaries emphasize different tails. At the supplied expression-grid rectangle, P07 lower-decile rates are about 4.40 and 4.30 times higher within 200 µm; the upper-tail ratios are 1.32 and 1.02. P08 upper-tail enrichment at that rectangle is 2.72. Thus “edge effect” cannot be equated with universally increased score, and a straight acquisition edge may cross biological tissue.','',
      'The available geometry shows zero partly off-image patches in NCBI785 and NCBI783, and 45 in NCBI784. Literal off-image padding therefore cannot explain the broad P08 boundary pattern. Partial tissue context, RNA coverage, composition and registration remain distinct possible contributors; this diagnostic does not determine their causal shares.','',
      '[All three H&E/boundary context pages](cell_linked_boundary_context.pdf), [P08 context](NCBI783_boundary_context.png), [P07 context](NCBI785_boundary_context.png), [ring profiles](ring_tail_profiles.png). The H&E views use saved registration and are not independent anatomical ground truth. The yellow contour is the tissue segmentation; the purple rectangle marks all input expression locations.','',
      '## What remains beyond boundaries?','',
      'The following diagnostic removes locations within each fixed width of **any** tissue-mask, image or input-grid boundary. It retains the original score and cutoffs; it does not create a fresh guaranteed decile inside the remainder.','',md(i[i['sample'].isin(samples)][['sample','band_um','fraction_retained','upper_10_retained','moran_all','moran_interior']]),'',
      'For P08, roughly 48% of the original top decile lies in the combined 200 µm boundary band, and 68% in the 400 µm band. Substantial interior spatial organization remains: Moran’s I is .699 overall, .677 after 200 µm exclusion and .571 after 400 µm exclusion. These values use the fixed 150 µm graph rule, rebuilt on retained points, and are descriptive comparisons rather than a new significance test.','',
      'A second diagnostic subtracts only the ten original hull-band means. In NCBI785/784/783 those means account for 3.18%, 1.34% and 5.99% of within-section score variance, respectively. Radius-graph Moran’s I changes .567→.561, .547→.549 and .699→.680. The percentage is variance explained by band means, not the fraction of spatial autocorrelation or technical artifact. This diagnostic is not a replacement score and is not the permutation test itself.','',md(r[r['sample'].isin(samples)][['sample','score_variance_fraction_between_ring_means','moran_radius150_original','moran_radius150_after_ring_mean_removal','archived_k6_ring_permutation_p']]),'',
      '## Consequence for the revision','',
      'Retain the conclusion of structured prediction error and explicitly credit the existing boundary-ring control. Add physical boundary context and the fixed-group erosion sensitivity to the spatial figure/supplement. Deciles can display strong boundary-associated patterns as well as interior regions, so visual concentration is insufficient to identify biological transition interfaces. Do not automatically regress out all boundary association: tissue margins can carry biology, and the current diagnostics do not identify a purely technical component.','',
      'This follow-up checks spatial score structure, not every gene/program/cell contrast after boundary restriction. A claim specifically about boundary-independent biological enrichment would require that additional population-specific analysis. The present figure plan does not make that stronger claim.','',
      '## Verification','',
      'All scored locations align to the Stage 1 native coordinates; signed mask/image distances have scalar checks; explicit edge sums independently reproduce graph Moran calculations; the original segment-distance formula agrees with the new hull geometry. Repairs to invalid contours are the previously documented in-memory repairs. Polygonal extraction removes only zero-area line remnants, with exactly zero symmetric-difference area. Two initial assertions caught an undefined geometry-collection boundary and floating-point polygon-area summation order; both failed logs are retained and no failed results are used.','',
      '[Protocol](PROTOCOL.md), [geometry/graph checks](checks.json), [original-ring checks](ring_checks.json), [all boundary effects](tail_boundary_enrichment.csv), [interior sensitivity](interior_sensitivity.csv).','']
    assert int((primary.relative_risk>1).sum())==26
    for f in ['checks.json','ring_checks.json']:assert json.loads((OUT/f).read_text())['status']=='pass'
    (OUT/'RESULTS.md').write_text('\n'.join(lines))
    (OUT/'summary.json').write_text(json.dumps(dict(status='complete',n_sections=32,n_sections_upper_decile_enriched_near_tissue_200um=26,n_idc_archived_ring_tests_p001=18,interpretation='Boundary enrichment coexists with retained interior and within-band spatial structure; cause unresolved; no score replacement'),indent=2)+'\n');print('Boundary report complete')


if __name__=='__main__':main()
