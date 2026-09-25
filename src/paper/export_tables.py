"""Export the numerical tables used in the current manuscript."""
from .paths import ANALYSIS_ROOT, CELL_ROOT
import pandas as pd

def table(name):
    return pd.read_csv(ANALYSIS_ROOT/'donors/tables'/(name+'.csv'))
def audit(name):
    return pd.read_csv(ANALYSIS_ROOT/name)
def oldtable(name):
    assert name == 'external_gene_features'
    frames=[]
    for family in ['coad','idc_visium']:
        frame=audit(f'external/{family}/revised_gene_features.csv')
        frame['family']=family
        frames.append(frame)
    return pd.concat(frames,ignore_index=True)
def primary_programs(frame):
    return frame[frame['tail'].eq(.25)&frame.grouping.eq('program_excluded')&frame.adjustment.eq('overlap_adjusted')]

def main():
    dest=ANALYSIS_ROOT/'Tables';dest.mkdir(exist_ok=True)
    def export(d,name):d.to_csv(dest/(name+'.csv'),index=False)
    reg=table('specimen_registry');export(reg,'supp_table_s15_specimen_registry');export(table('fold_registry'),'supp_table_s15_fold_registry')
    q=table('model_gene_cohort_quality');r=q[q.regressor.eq('ridge')].pivot(index=['cohort','gene'],columns='encoder',values='pearson').reset_index();r['mean_pearson']=r[['uni','virchow2','hoptimus0']].mean(axis=1)
    f=table('gene_features');r=r.merge(f.drop(columns='mean_pearson'),on=['cohort','gene'],validate='one_to_one');export(r,'supp_table_s1_per_gene_performance');export(q,'supp_table_s1_all_model_performance')
    export(table('model_donor_quality'),'supp_table_s1_donor_prediction_quality');export(table('donor_quartile_quality'),'supp_table_s1_quartile_prediction_quality');export(table('gene_donor_quality'),'supp_table_s1_gene_quartile_quality')
    export(table('bridge_genes'),'supp_table_s2_bridge_genes')
    for source,suffix in [('gene_section_effects','section'),('gene_donor_effects','donor'),('gene_cohort_effects','cohort'),('gene_split_group_sensitivity','split_sensitivity')]:export(table(source),'supp_table_s3_'+suffix+'_effects')
    c=table('gene_cohort_effects');c=c[c.metric.eq('standardized_effect')]
    a=c[c.grouping.eq('full')&c.adjustment.eq('unadjusted')].drop(columns=['grouping','adjustment']);b=c[c.grouping.eq('gene_excluded')&c.adjustment.eq('unadjusted')].drop(columns=['grouping','adjustment']);z=c[c.grouping.eq('gene_excluded')&c.adjustment.eq('overlap_adjusted')].drop(columns=['grouping','adjustment'])
    keys=['cohort','gene','outcome','metric'];a=a.merge(b,on=keys,suffixes=('_full','_excluded')).merge(z[keys+['mean']],on=keys).rename(columns={'mean':'mean_excluded_adjusted'});a['excluded_minus_full']=a.mean_excluded-a.mean_full;a['adjusted_minus_excluded']=a.mean_excluded_adjusted-a.mean_excluded;export(a,'supp_table_s3_attenuation')
    export(table('predictability_expression_relationship'),'supp_table_s3_predictability_relationship')
    pairs=audit('contrasts/patient_profile_pairs.csv');pairs=pairs[pairs.family.eq('10x_janesick')];export(pairs,'supp_table_s4_validation_profile_pairs')
    summary=audit('contrasts/patient_profile_summary.csv');summary=summary[summary.family.eq('10x_janesick')].drop(columns='primary_holm_p');export(summary,'supp_table_s4_grouping_reference')
    # Discovery grouping analysis is withdrawn, not re-labeled as donor repeatability.
    export(audit('programs/coverage.csv'),'supp_table_s5_pathway_overlap')
    er=r[['cohort','gene','uni','virchow2','hoptimus0']].copy();er['range']=er[['uni','virchow2','hoptimus0']].max(axis=1)-er[['uni','virchow2','hoptimus0']].min(axis=1);export(er,'supp_table_s6_encoder_variability')
    halves=audit('reliability/gene_half_stability.csv');halves['patient']=halves['sample'].map(reg.set_index('sample').patient);export(halves,'supp_table_s7_dual_track')
    export(audit('reliability/spatial_structure.csv').drop(columns='patient'),'supp_table_s8_spatial_structure')
    export(audit('boundary/ring_diagnostic.csv').drop(columns='unit'),'supp_table_s8_ring_diagnostic')
    for name in ['program_section_effects','program_unit_effects','program_cohort_effects','program_split_group_sensitivity']:export(table(name),'supp_table_s9_'+name)
    cut=table('program_section_effects');cut=cut[cut.grouping.eq('program_excluded')&cut.adjustment.eq('overlap_adjusted')]
    cut=cut.groupby(['family','unit','pathway','tail','outcome']).section_standardized_effect.mean().reset_index()
    cut=cut.groupby(['family','pathway','tail','outcome']).section_standardized_effect.mean().reset_index();export(cut,'supp_table_s9_fixed_scale_cutoff_summary')
    p=primary_programs(table('program_cohort_effects'));p=p[p.metric.eq('standardized_effect')]
    export(oldtable('external_gene_features').query('family=="coad"'),'supp_table_s10_coad_gene_features')
    export(table('program_section_effects').query('family=="coad"'),'supp_table_s11_coad_programs')
    export(table('program_cohort_effects').query('family=="coad"'),'supp_table_s12_coad_consistency')
    export(table('external_gene_pairs'),'supp_table_s13_cross_tissue_predictability');export(table('external_program_comparisons'),'supp_table_s13_exact_program_comparisons')
    for name in ['cell_mixture','whole_spot_effects','within_lineage_effects','support']:export(audit('composition/'+name+'.csv'),'supp_table_s14_'+name)
    export(audit('contrasts/tumor_rich_emt_effects.csv'),'supp_table_s14_tumor_rich_emt');export(audit('contrasts/tumor_rich_emt_support.csv'),'supp_table_s14_tumor_rich_support');export(audit('genes/marker_coverage.csv'),'supp_table_s14_marker_coverage')
    canonical=['VIM','CDH2','FN1','SNAI1','SNAI2','ZEB1','ZEB2']
    g=table('gene_cohort_effects');export(g[g.gene.isin(canonical)],'supp_table_s14_canonical_emt_effects')
    g=table('gene_donor_quality');export(g[g.gene.isin(canonical)],'supp_table_s14_canonical_emt_prediction_quality')
    for name in ['boundary_section_effects','boundary_unit_effects','boundary_cohort_effects']:export(table(name),'supp_table_s8_'+name)
    export(table('gene_feature_metrics'),'supp_table_s1_feature_model_metrics');export(table('gene_feature_coefficients'),'supp_table_s1_feature_coefficients');export(table('gene_feature_cv_predictions'),'supp_table_s1_feature_cv');export(table('gene_feature_influence'),'supp_table_s1_feature_donor_influence')
    export(audit('matching/matching_diagnostics.csv').drop(columns='patient'),'supp_table_s3_morphology_overlap')
    export(pd.read_csv(CELL_ROOT/'emt_spatial_enrichment_v1/results/statistics.csv'),'supp_table_s16_spatial_statistics')
    export(pd.read_csv(CELL_ROOT/'emt_spatial_enrichment_v1/results/covariate_balance.csv'),'supp_table_s16_covariate_balance')

    export(pd.read_csv(CELL_ROOT/'emt_cells_v1/results/source_label_evidence.csv')[['sample','patient','label','source_group','n_cells','n_tf_epithelial_coexpression','fraction_tf_epithelial_coexpression','n_nuclear_coexpression','median_transcripts']],'supp_table_s16_source_cell_counts')

    export(audit('external/common_member_coverage.csv'),'visium_common_member_coverage')
    export(oldtable('external_gene_features').query("family=='idc_visium'"),'visium_gene_features')
    export(audit('external/quartile_ordering.csv'),'visium_quartile_ordering')
    export(audit('external/specimen_prediction_quality.csv'),'visium_specimen_prediction_quality')

if __name__ == '__main__':
    main()
