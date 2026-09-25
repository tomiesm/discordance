"""Curate verified inputs for figures; preserve definitions and exclude dropped analyses."""
from common import *


def main():
    raw=original('stage_02_prediction/patient_model_metrics.csv');table(raw,'model_patient_quality')
    table(raw[raw.regressor.eq('ridge')].groupby(['cohort','patient']).mean(numeric_only=True).reset_index(),'ridge_patient_quality')
    table(original('stage_02_prediction/cohort_patient_mean_metrics.csv'),'model_cohort_quality')
    table(original('stage_02_prediction/gene_model_metrics.csv'),'model_gene_quality')
    qp=original('stage_07_gene_audit/quartile_profiles.csv');table(qp,'gene_quartile_profiles')
    section=qp.groupby(['cohort','patient','sample','quartile'])[['absolute','baseline_mae','observed_log','signed']].mean().reset_index();table(section,'section_quartile_quality')
    patient=section.groupby(['cohort','patient','quartile'])[['absolute','baseline_mae','observed_log','signed']].mean().reset_index();patient['relative_mae_gain']=1-patient.absolute/patient.baseline_mae;table(patient,'patient_quartile_quality')
    for source,dest in [('prediction_quality','gene_section_quality'),('patient_prediction_quality','gene_patient_quality'),('patient_effects','gene_patient_effects'),('cohort_effects','gene_cohort_effects'),('all_bridge_gene_comparisons','bridge_genes'),('marker_coverage','marker_coverage'),('named_marker_evidence','marker_evidence'),('gene_exclusion_attenuation','gene_exclusion_attenuation'),('predictability_effect_relationship','predictability_vs_expression')]:table(original(f'stage_07_gene_audit/{source}.csv'),dest)
    programs=[];summaries=[];agree=[];locations=[];quality=[];section_programs=[]
    for family in FAMILIES:
        base=f'stage_10_decile_sensitivity/{family}'
        a=original(base+'/unit_program_effects.csv');programs.append(a[a['tail'].isin([.2,.25,.3])])
        a=original(base+'/cohort_program_summary.csv');summaries.append(a[a['tail'].isin([.2,.25,.3])])
        a=original(base+'/section_program_effects.csv');section_programs.append(a[a['tail'].isin([.2,.25,.3])])
        a=original(base+'/encoder_stability.csv');agree.append(a[a['tail'].isin([.2,.25,.3])])
        a=original(base+'/tail_quality.csv');quality.append(a[a['tail'].isin([.2,.25,.3])])
        a=pd.read_parquet(track(AUDIT/base/'locations.parquet'));locations.append(a)
    table(pd.concat(programs,ignore_index=True),'program_unit_effects');table(pd.concat(summaries,ignore_index=True),'program_cohort_effects');table(pd.concat(section_programs,ignore_index=True),'program_section_effects')
    table(pd.concat(agree,ignore_index=True),'encoder_agreement');table(pd.concat(quality,ignore_index=True),'tail_quality')
    loc=pd.concat(locations,ignore_index=True);loc.to_parquet(OUT/'tables/locations.parquet',index=False)
    for source,dest in [('gene_half_stability','half_gene_stability'),('spatial_structure','spatial_structure'),('graph_diagnostics','graph_diagnostics')]:table(original(f'stage_04_reliability/{source}.csv'),dest)
    table(original('stage_05_biology/coverage.csv'),'idc_program_members');table(original('stage_05_biology/quartile_profiles.csv'),'program_quartile_profiles')
    for source in ['matching_diagnostics','matching_covariate_balance']:table(original(f'stage_06_inference/{source}.csv'),source)
    for source in ['cell_mixture','whole_spot_effects','within_lineage_effects','support']:table(original(f'stage_08_cell_composition/{source}.csv'),'cell_'+source)
    zones=original('stage_10_decile_sensitivity/zone_contrasts.csv');table(zones[zones['tail'].eq(.25)],'cell_neighborhood_quartiles')
    zd=original('stage_10_decile_sensitivity/zone_descriptives.csv');table(zd[zd['tail'].eq(.25)],'cell_neighborhood_descriptives')
    for name in ['model_metrics','coefficients','cv_predictions','fold_diagnostics','annotation_audit','patient_influence']:table(original(f'stage_11_gene_feature_model/{name}.csv'),'gene_feature_'+name)
    fs=[]
    for family in IDC:
        a=original(f'stage_11_gene_feature_model/{family}/revised_features.csv');a['cohort']=family;fs.append(a)
    table(pd.concat(fs,ignore_index=True),'gene_features')
    for name in ['specimen_prediction_quality','gene_predictability_pairs','gene_predictability_comparisons','program_comparisons','program_comparison_summary','common_member_coverage','quartile_ordering']:table(original(f'stage_09_external_replication/{name}.csv'),'external_'+name)
    fs=[];qs=[];cs=[]
    for family in FAMILIES[2:]:
        a=original(f'stage_09_external_replication/{family}/revised_gene_features.csv');a['family']=family;fs.append(a)
        a=original(f'stage_09_external_replication/{family}/quartile_quality.csv');a['family']=family;qs.append(a)
        a=original(f'stage_09_external_replication/{family}/program_coverage.csv');a['family']=family;cs.append(a)
    table(pd.concat(fs,ignore_index=True),'external_gene_features');table(pd.concat(qs,ignore_index=True),'external_quartile_quality');table(pd.concat(cs,ignore_index=True),'external_program_members')
    table(original('stage_12_boundary_diagnostic/ring_diagnostic.csv'),'ring_diagnostic')
    a=original('stage_12_boundary_diagnostic/interior_sensitivity.csv');table(a.drop(columns=[c for c in a.columns if '_10_' in c]),'boundary_interior')
    a=original('stage_12_boundary_diagnostic/tail_boundary_enrichment.csv');table(a[a['tail'].eq(.25)],'boundary_quartile_enrichment')
    b=pd.read_parquet(track(AUDIT/'stage_12_boundary_diagnostic/spot_boundary_diagnostics.parquet'));b.to_parquet(OUT/'tables/boundary_locations.parquet',index=False)
    coverage=original('stage_01_provenance/coverage_summary.csv');table(coverage,'patch_coverage');table(original('stage_01_provenance/idc_coverage_distributions.csv'),'coverage_expression_summary')
    manifest=[]
    for (family,sample,unit),ss in loc.groupby(['family','sample','unit']):
        c=coverage[coverage['sample'].eq(sample)].iloc[0]
        genes=json.loads(track(REPO/(f'data/v3/gene_list_{family}.json' if family in IDC else f'outputs/{family}/gene_panel.json')).read_text())
        manifest.append(dict(family=family,sample=sample,patient_or_known_group=unit,platform='Visium' if family=='idc_visium' else 'Xenium',input_locations=int(c.raw_spots),analyzed_locations=len(ss),modeled_genes=len(genes),patch_width_um=c.patch_footprint_um,expression_grid_spacing_um=100 if family!='idc_visium' else np.nan,partial_image_patches=int(c.retained_partially_outside_slide),note='P07 paired platform' if sample=='NCBI776' else 'Joint Block A exclusion' if sample in ['TENX13','TENX14'] else 'Other donor links unresolved' if family=='idc_visium' else ''))
    table(pd.DataFrame(manifest),'T1_sample_manifest')
    models=read('gene_feature_model_metrics');table(models[models.version.eq('revised_all_sections')],'T2_gene_feature_models')
    table(pd.DataFrame(manifest).query("family=='idc_visium'"),'TV1_visium_manifest')
    profiles=pd.read_parquet(track(AUDIT/'stage_03_score/spot_score_diagnostics.parquet'));profiles.to_parquet(OUT/'tables/score_definition.parquet',index=False)
    assert len(loc)==sum(x['analyzed_locations'] for x in manifest)
    assert len(manifest)==32
    for a in [pd.concat(programs),pd.concat(summaries),pd.concat(agree),pd.concat(quality),zones[zones['tail'].eq(.25)]]:assert not a['tail'].eq(.1).any()
    flush('prepare');print('Curated quartile figure inputs complete',flush=True)


if __name__=='__main__':main()
