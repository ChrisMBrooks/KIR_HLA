CREATE TABLE 
    KIR_HLA_STUDY.raw_immunophenotype_measurement (
        subject_id INTEGER NOT NULL,
        measurement_id VARCHAR(15) NOT NULL,
        measurement_value DECIMAL(20,10),
        PRIMARY KEY (subject_id, measurement_id)
        ) 
    PARTITION BY HASH(subject_id)
    PARTITIONS 100;

CREATE TABLE 
    KIR_HLA_STUDY.immunophenotype_assay (
        numeric_id INTEGER NOT NULL AUTO_INCREMENT,
        measurement_id VARCHAR(15) NOT NULL,
        PRIMARY KEY (numeric_id)
    );

INSERT INTO 
    KIR_HLA_STUDY.immunophenotype_assay(measurement_id) 
SELECT 
	DISTINCT(measurement_id) 
FROM 
	KIR_HLA_STUDY.raw_immunophenotype_measurements;

CREATE TABLE 
    KIR_HLA_STUDY.train_immunophenotype_measurement 
LIKE 
    KIR_HLA_STUDY.raw_immunophenotype_measurement;


INSERT INTO KIR_HLA_STUDY.validation_partition (public_id, validation_partition, subject_id)
SELECT 
	KIR_HLA_STUDY.validation_partition.public_id, 
    KIR_HLA_STUDY.validation_partition.validation_partition, 
    KIR_HLA_STUDY.raw_public_mapping.flow_jo 
FROM 
	KIR_HLA_STUDY.validation_partition
INNER JOIN 
	KIR_HLA_STUDY.raw_public_mapping 
ON 
	KIR_HLA_STUDY.validation_partition.public_id = 
    KIR_HLA_STUDY.raw_public_mapping.public_id
ON DUPLICATE KEY UPDATE subject_id = flow_jo


INSERT INTO KIR_HLA_STUDY.val_immunophenotype_measurement(subject_id, measurement_id, measurement_value)
SELECT 
	KIR_HLA_STUDY.raw_immunophenotype_measurement.subject_id as subject_id, 
    KIR_HLA_STUDY.raw_immunophenotype_measurement.measurement_id, 
    KIR_HLA_STUDY.raw_immunophenotype_measurement.measurement_value
FROM 
	KIR_HLA_STUDY.raw_immunophenotype_measurement
INNER JOIN 
	KIR_HLA_STUDY.validation_partition 
ON 
	KIR_HLA_STUDY.validation_partition.subject_id = 
    KIR_HLA_STUDY.raw_immunophenotype_measurement.subject_id
WHERE KIR_HLA_STUDY.validation_partition.validation_partition = 'VALIDATION'

CREATE VIEW 
	KIR_HLA_STUDY.public_mapping_vw AS
SELECT 
	public_id, flow_jo "subject_id" 
FROM 
	KIR_HLA_STUDY.raw_public_mapping;

CREATE TABLE 
    KIR_HLA_STUDY.model_result_elastic_net_primary_coeff (
		coeff FLOAT NOT NULL, 
        feature_name VARCHAR(25) NOT NULL, 
        relevance_cut_off FLOAT NOT NULL, 
        predictor_id VARCHAR(15) NOT NULL, 
		alpha FLOAT NOT NULL,
		l1_ratio FLOAT NOT NULL, 
		measured_abs_error FLOAT NOT NULL, 
		grid_search_alpha_range VARCHAR(25),
		grid_search_l1_ratio_range VARCHAR(25), 
        cross_val_n_splits INTEGER NOT NULL,
		cross_val_n_repeats INTEGER NOT NULL, 
        run_id CHAR(32) NOT NULL,
        PRIMARY KEY (feature_name, run_id)
	)

CREATE TABLE 
    KIR_HLA_STUDY.hla_allele (
        ebi_id VARCHAR(15) NOT NULL,
        ebi_name VARCHAR(20) NOT NULL,
        hla_gene VARCHAR(1) NOT NULL,
        short_code INTEGER NOT NULL,
        protein_sequence TEXT,
        PRIMARY KEY (ebi_id)
    );


 CREATE TABLE KIR_HLA_STUDY.functional_kir_genotype(
    public_id VARCHAR(15) NOT NULL,
    kir2dl1 TINYINT(1) NOT NULL,
    kir2dl2 TINYINT(1) NOT NULL,
    kir2dl3 TINYINT(1) NOT NULL,
    kir3dl1 TINYINT(1) NOT NULL,
    hla_c_c1 TINYINT(1) NOT NULL,
    hla_c_c2 TINYINT(1) NOT NULL,
    hla_b_46_c1 TINYINT(1) NOT NULL,
    hla_b_73_c1 TINYINT(1) NOT NULL,
    hla_b_bw4 TINYINT(1) NOT NULL,
    f_kir2dl1 TINYINT(1) NOT NULL,
    f_kir2dl2_s TINYINT(1) NOT NULL,
    f_kir2dl2_w TINYINT(1) NOT NULL,
    f_kir2dl3 TINYINT(1) NOT NULL,
    f_kir3dl1 TINYINT(1) NOT NULL,
    f_kir_count INTEGER NOT NULL,
    PRIMARY KEY (public_id)
)

CREATE TABLE KIR_HLA_STUDY.model_result_ols (
        feature_name VARCHAR(25) NOT NULL, 
		beta_0 FLOAT NOT NULL, 
		beta_1 FLOAT NOT NULL, 
		beta_2 FLOAT NOT NULL, 
		beta_3 FLOAT NOT NULL, 
		beta_4 FLOAT NOT NULL, 
		beta_5 FLOAT NOT NULL, 
        run_id CHAR(32) NOT NULL,
        PRIMARY KEY (feature_name, run_id)
	)

    CREATE TABLE KIR_HLA_STUDY.model_result_el_net(
        run_id CHAR(32) NOT NULL, 
        mae FLOAT NOT NULL, 
        non_zero_coeff_count INTEGER NOT NULL, 
        l1_ratio FLOAT NOT NULL, 
        l1_ratio_min FLOAT NOT NULL, 
        l1_ratio_max FLOAT NOT NULL, 
        l1_ratio_step FLOAT NOT NULL, 
        alpha FLOAT NOT NULL, 
        alpha_min FLOAT NOT NULL, 
        alpha_max FLOAT NOT NULL, 
        alpha_step FLOAT NOT NULL
    )

    CREATE TABLE KIR_HLA_STUDY.model_result_rand_forest(
        run_id CHAR(32) NOT NULL, 
        mae FLOAT NOT NULL, 
        max_nodes FLOAT NOT NULL,
        max_nodes_min FLOAT NOT NULL,
        max_nodes_max FLOAT NOT NULL,
        max_nodes_step FLOAT NOT NULL,
        num_trees FLOAT NOT NULL,
        num_trees_min FLOAT NOT NULL,
        num_trees_max FLOAT NOT NULL
    )

    CREATE TABLE KIR_HLA_STUDY.model_result_el_net_coeffs(
        run_id CHAR(32) NOT NULL, 
        phenotype_label VARCHAR(15) NOT NULL, 
        beta FLOAT NOT NULL,
        l1_ratio FLOAT NOT NULL,
        alpha FLOAT NOT NULL,
        PRIMARY KEY (run_id, phenotype_label)
    )

    CREATE TABLE KIR_HLA_STUDY.immunophenotype_definitions(
        phenotype_id VARCHAR(50) NOT NULL, 
        marker_definition VARCHAR(50) NOT NULL, 
        parent_population VARCHAR(50) NOT NULL, 
        PRIMARY KEY (phenotype_id)
    )

SELECT *
FROM KIR_HLA_STUDY.model_result_el_net_coeffs
INNER JOIN KIR_HLA_STUDY.immunophenotype_definitions
ON KIR_HLA_STUDY.model_result_el_net_coeffs.phenotype_label = KIR_HLA_STUDY.immunophenotype_definitions.phenotype_id;