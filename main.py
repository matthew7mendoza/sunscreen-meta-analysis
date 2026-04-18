import scripts.utilities as ut
import scripts.analyzer as an
import scripts.visuals as vis

def main():
    print("\n")
    print("SUNSCREEN APPLICATION META-ANALYSIS: ")
    print("\n")

    # initialize and prepare dataset.
    rob_file_path = "data/rob_judgements.json"
    dataset = ut.MetaAnalysisDataset()
    dataset.load_from_csv("data/extracted_papers.csv")
    dataset.calculate_dynamic_cvs(fallback_cv = 0.25)
    dataset.apply_imputations() 
    my_rob_judgments = dataset.load_rob_judgements(rob_file_path)


    # recommended assumptions, per Cochrane Handbook for Systematic Review of Interventions
    r_assumptions = [0.2, 0.5, 0.8] 
    
    # target --> primary data, meaning r of 0.5
    target_forest_data = None
    target_forest_pooled = None

    for r_test in r_assumptions:
        print(f"\n" + "-"*50)
        print(f" TESTING IMPUTED WITHIN-SUBJECT CORRELATION: r = {r_test}")
        print("-" * 50)
        
        # calculate study effect sizes for r-value in r_assumptions
        iteration_results = an.run_pipeline(dataset.records, assumed_r = r_test)
        study_data_pairs = list(zip(iteration_results, dataset.records))

        # primary analysis, r = 0.5, including all studies.
        primary_groups = ut.AnalysisReporter.group_and_filter(study_data_pairs)
        ut.AnalysisReporter.print_results("Primary Analysis: Test Type & Metric", primary_groups)
        
        # hold findings for plotting primary analysis
        if r_test == 0.5 and "IN_VIVO | MG/CM2" in primary_groups:
            target_forest_data = primary_groups["IN_VIVO | MG/CM2"]
            target_forest_pooled = an.pool_subgroup(target_forest_data)


        # sensitivity analysis 1, excluding skewed data
        non_skewed = ut.AnalysisReporter.group_and_filter(study_data_pairs, lambda r: not r.metadata.is_skewed)
        ut.AnalysisReporter.print_results("Sensitivity Analysis: Excluding Skewed IQR Data", non_skewed)

        # sensitivity analysis 2: exclude imputed standard deviations
        no_cv = ut.AnalysisReporter.group_and_filter(study_data_pairs, lambda r: not r.metadata.used_cv_imputation)
        ut.AnalysisReporter.print_results("Sensitivity Analysis: Excluding CV-Imputed SDs", no_cv)

    # generate visualizations
    print("\n" + "="*60)
    print(" GENERATING COCHRANE VISUALIZATIONS ")
    print("="*60)

    # PRISMA flowchart
    vis.PRISMAFlowchart().draw()

    if target_forest_data and target_forest_pooled:
        vis.ForestPlotVisualizer().draw(
            target_forest_data, 
            target_forest_pooled, 
            title = "Sunscreen Applied Effect (r = 0.5)\nIN_VIVO | MG/CM2",
            save_filename = "forest_plot_r05.svg"
        )

    rob_vizual = vis.RiskOfBiasVisualizer(my_rob_judgments)
    rob_vizual.draw_traffic_light_plot()
    rob_vizual.draw_summary_bar_chart()
    
    print("\ncomplete. figures saved to the 'figures' directory.")

if __name__ == "__main__":
    main()

