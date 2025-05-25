# library(devtools)
# install_github("michaelyanwang/dcalasso")

library(survival)
library(dcalasso)
library(profvis)
library(rms)
library(glmnet)

# Helper functions
dac_BIC <- function(cph_mod) {
  return(cph_mod$loglik / cph_mod$n + log(cph_mod$nevent) * length(cph_tv$coefficients))
}

concordance_index <- function(mod, data, time_col="stop", event_col="event", cph=TRUE) {
  time <- data[, time_col]
  status <- data[, event_col]
  
  # Survival package automatically centers the linear predictors
  if (cph) {
    x <- predict(mod, newdata=data)
  } else {
    x <- scale(predict(mod, newdata=data)$fit, scale=F)
  }
  
  n <- length(time)
  ord <- order(time, -status)
  time <- time[ord]
  status <- status[ord]
  x <- x[ord]
  
  wh <- which(status == 1)
  total <- concordant <- 0
  
  # Loop over event rows
  for (i in wh) {
    # Get indices for all subjects that have later times (time[j] > time[i])
    later_times <- which(time > time[i])
    
    # Get the predictions for those subjects
    x_later <- x[later_times]
    x_i <- x[i]
    
    # Compare predictions for concordance
    concordant <- concordant + sum(x_later < x_i) + 0.5 * sum(x_later == x_i)
    total <- total + length(later_times)
  }
  
  # Calculate concordance
  concordance <- concordant / total
  return(concordance)
}


scale_data <- function(df_train, df_test=NULL, cont_var=c(), wide=F, tv=F, tv_cont_var=c("hba1c", "GFR", "kolesterol", "systoliskt", "bmi")) {
  if (!wide) {
    # Standard operations
    if (!tv) {
      # Time-independent
      train_means <- colMeans(df_train[, cont_var])
      train_sd <- apply(df_train[, cont_var], 2, sd)
      
      cat("Means:\n", train_means,
          "\n\nSDs:\n", train_sd)
      
      df_train[, cont_var] <- scale(df_train[, cont_var], center=train_means, scale=train_sd)
      
      if (!is.null(df_test)) {
        df_test[, cont_var] <- scale(df_test[, cont_var], center=train_means, scale=train_sd)  
      }  
    } else {
      # Time-dependent
      train_means <- colMeans(df_train[df_train$start == 0, cont_var])
      train_sd <- apply(df_train[df_train$start == 0, cont_var], 2, sd)
      
      cat("Means:\n", train_means,
          "\n\nSDs:\n", train_sd)
      
      df_train[, cont_var] <- scale(df_train[, cont_var], center=train_means, scale=train_sd)
      
      if (!is.null(df_test)) {
        df_test[, cont_var] <- scale(df_test[, cont_var], center=train_means, scale=train_sd)  
      }
    }
    
  } else {
    train_means_const <- colMeans(df_train[, setdiff(cont_var, tv_cont_var)])
    train_sd_const <- apply(df_train[, setdiff(cont_var, tv_cont_var)], 2, sd)
    
    df_train[, setdiff(cont_var, tv_cont_var)] <- scale(df_train[, setdiff(cont_var, tv_cont_var)], center=train_means_const, scale=train_sd_const)
    
    for (tv_var in tv_cont_var) {
      # Scale tv columns according to overall mean/sd
      cols_to_scale_train <- grep(paste0("^", tv_var, "_\\d+$"), names(df_train), value=T)
      
      values <- unlist(df_train[cols_to_scale_train])
      overall_mean <- mean(values, na.rm=T)
      overall_sd <- sd(values, na.rm=T)
      
      df_train[cols_to_scale_train] <- lapply(df_train[cols_to_scale_train], function(x) (x - overall_mean) / overall_sd)
      
      if (!is.null(df_test)) {
        cols_to_scale_test <- grep(paste0("^", tv_var, "_\\d+$"), names(df_test), value=T)
        df_test[cols_to_scale_test] <- lapply(df_test[cols_to_scale_test], function(x) (x - overall_mean) / overall_sd)
      }
      
    }
  }
  
  return(list(train=df_train, test=df_test))
}


load_data <- function(path_train, path_test="", cont_var=c(), tv=T, wide=F) {
  df_train <- read.csv(path_train)
  
  if (path_test == "") {
    df_test <- NULL
  } else {
    df_test <- read.csv(path_test)  
  }
  
  scale_res <- scale_data(df_train, df_test, cont_var, wide, tv)
  df_train <- scale_res$train
  
  df_train[df_train == "True"] <- 1
  df_train[df_train == "False"] <- 0
  
  if (!is.null(df_test)) {
    df_test <- scale_res$test
    
    df_test[df_test == "True"] <- 1
    df_test[df_test == "False"] <- 0
  }
  
  if (tv) {
    df_train["event"] <- as.integer(df_train$event)
    df_train <- df_train[df_train$stop > df_train$start,]
    
    if (!is.null(df_test)) {
      df_test["event"] <- as.integer(df_test$event)
      df_test <- df_test[df_test$stop > df_test$start,]
    }
  } else {
    df_train[c("death", "ami", "stroke")] <- lapply(df_train[c("death", "ami", "stroke")], as.integer)
    
    if (!is.null(df_test)) {
      df_test[c("death", "ami", "stroke")] <- lapply(df_test[c("death", "ami", "stroke")], as.integer) 
    }
  }
  
  return(list(train=df_train, test=df_test))
}


# fit_cox <- function(df_train, df_test, form, cont_var){
#   x_var <- all.vars(form[[3]])
#   cat_var <- setdiff(x_var, cont_var)
#   rcs_var <- paste0("rcs_", cont_var)
#   new_var <- c()
#   
#   time_col <- all.vars(form[[2]])[1]
#   event_col <- all.vars(form[[2]])[2]
#   
#   # Build X matrix with manually created rcs columns
#   X <- df_train[, cat_var]
#   X_test <- df_test[, cat_var]
#   for (var in cont_var) {
#     rcs_var <- rcspline.eval(df_train[, var], nk=5, inclx=T)
#     rcs_var_test <- rcspline.eval(df_test[, var], nk=5, inclx=T)
#     
#     colnames(rcs_var) <- paste0("rcs_", var, "_", seq_len(ncol(rcs_var)))
#     colnames(rcs_var_test) <- paste0("rcs_", var, "_", seq_len(ncol(rcs_var_test)))
#     
#     new_var <- append(new_var, colnames(rcs_var))
#     
#     X <- cbind(X, rcs_var)
#     X_test <- cbind(X_test, rcs_var_test)
#   }
#   
#   # Add interaction columns where necessary
#   term_labels <- attr(terms(form), "term.labels")
#   int_terms <- term_labels[grepl(":", term_labels)]
#   if (length(int_terms) > 0) {
#     for (int_term in int_terms) {
#       # Extract which variables are in the interaction
#       vars <- unlist(strsplit(int_term, ":"))
#       
#       if (grepl("rcs\\(", vars[1])) {
#         var1 <- gsub("rcs\\(|\\)", "", vars[1])  
#       }
#       
#       if (grepl("rcs\\(", vars[2])) {
#         var2 <- gsub("rcs\\(|\\)", "", vars[2])  
#       }
#       
#       cols1 <- new_var[grepl(var1, new_var)]
#       cols2 <- new_var[grepl(var2, new_var)]
#       
#       for (c1 in cols1) {
#         for (c2 in cols2) {
#           int_var <- as.data.frame(X[, c1] * X[, c2])
#           colnames(int_var) <- paste0(c1, "_X_", c2)
#           
#           new_var <- append(new_var, paste0(c1, "_X_", c2))
#           X <- cbind(X, int_var)
#         }
#       }
#       
#     }
#   }
#   new_var <- append(new_var, cat_var)
#   
#   y <- Surv(df_train[, time_col], df_train[, event_col])
#   cv_fit <- cv.glmnet(as.matrix(X), y, family="cox")
#   coef_cv <- coef(cv_fit, s="lambda.min")
#   
#   # Turn into CPH object
#   data <- cbind(X, df_train[, c(time_col, event_col)])
#   data_test <- cbind(X_test, df_test[, c(time_col, event_col)])
#   
#   cph <- coxph(as.formula(paste0("Surv(", time_col, ", ", event_col, ") ~ ", paste(new_var, collapse="+"))), data, x=T, init=as.numeric(coef_cv), iter.max=0)
#   
#   conc_train <- concordance(cph, newdata=data)$concordance
#   conc_test <- concordance(cph, newdata=data_test)$concordance
#   
#   return(list(cph=cph, conc_train=conc_train, conc_test=conc_test))
# }


fit_cox <- function(df_train, df_test, form, gr_i){
  # Define null model and variable scope
  null_form <- as.formula(paste0(deparse(form[[2]]), " ~ factor(cpap_treated)"))
  scope <- as.formula(paste0("~ ", paste(deparse(form[[3]]), collapse="")))
  # scope <- update(scope, . ~ . - factor(cpap_treated))
  null_mod <- coxph(null_form, df_train)
  
  # Compute and evaluate
  step_mod <- stepAIC(null_mod, scope=list(lower=as.formula("~ factor(cpap_treated)"), upper=scope), direction="both")
  
  # Save model to file
  # saveRDS(step_mod, paste0("Models/cph_", gr_i, ".rds"))
  
  conc_train <- concordance(step_mod, newdata=df_train)$concordance
  conc_test <- concordance(step_mod, newdata=df_test)$concordance
  
  return(list(cph=step_mod, conc_train=conc_train, conc_test=conc_test))
}


fit_dac <- function(df_train, form, gamma_values=seq(0.5, 2, by=0.5), K=20) {
  mods <- list()
  bics <- c()
  
  j <- 1
  for (i in seq_along(gamma_values)) {
    gamma <- gamma_values[i]
    penalties <- c(0, rep(gamma, length(attr(terms(form), "term.labels")) - 1))
    dac <- dcalasso(form, data=df_train, gamma=penalties, K=K)
    
    mods[[j]] <- dac
    bics <- c(bics, dac$BIC.opt)
    
    j <- j + 1
  }
  
  dac_opt <- mods[[which(bics == min(bics))]]
  
  return(dac_opt)
}


run_models <- function(vec_forms, df_train, df_test, time_col, event_col, cont_var, prof_out_file, gr_i, K=20, bigd=T) {
  assign("df_train", df_train, envir=.GlobalEnv)
  
  i <- 1
  for (form in vec_forms) {
    print(paste("### FORMULA", i, "###"))
    
    # Train models
    prof <- profvis({
      cph_list <- fit_cox(df_train, df_test, form, cont_var)
      cph <- cph_list$cph
      conc_train <- cph_list$conc_train
      conc_test <- cph_list$conc_test
      
      if (bigd) {
        try({
          dac <- fit_dac(df_train, form, K=K)
          
          print("--- DAC RESULTS ---")
          print(summary(dac))
          conc_train_dac <- concordance_index(dac, df_train, time_col, event_col, cph=F)
          conc_test_dac <- concordance_index(dac, df_test, time_col, event_col, cph=F)
        }, silent=T
        )
      }
      
      # Evaluations
      zph <- cox.zph(cph)
      
      # Print summaries and plot
      print("--- SURVIVAL RESULTS ---")
      print(summary(cph))
      
      plot(zph, resid=F)
      termplot(cph)
      
      # Compute concordance index
      print("--- CONCORDANCE RESULTS ---")
      cat("C-index train:", conc_train,
          "\nC-index test:", conc_test
      )
      
      if (bigd) {
        try({
          cat("\nC-index DAC train:", conc_train_dac,
              "\nC-index DAC test:", conc_test_dac)
        }, silent=T)
      }
      
    })
    
    # Save profiling results
    htmlwidgets::saveWidget(prof, paste0("Profiling/dac_form", i, "_", prof_out_file))
    
    i <- i + 1
  }
}


generate_formulas <- function () {
  base_form <- . ~ factor(cpap_treated) + rcs(alder) + factor(sex) + rcs(bmi) + factor(rokare) + factor(ami_history) + factor(stroke_history) + factor(Civil) + factor(HushallsType_RTB) + factor(Sun2000niva_old) + rcs(DispInkKEHB04) + rcs(Raks_AndelArblosInk) + rcs(Raks_AndelSjukInk) + rcs(Raks_AndelEkBisInk) + factor(FodelseLand_EU27_2020) + factor(antithrombotic_agents) + factor(antihypertensive_comb) + factor(lipid_modifying_agents) + factor(cpap_treated):factor(antithrombotic_agents) + factor(cpap_treated):factor(antihypertensive_comb) + factor(cpap_treated):factor(lipid_modifying_agents) + rcs(alder):rcs(bmi) 
  
  list_forms <- list()
  
  for (time_setting in c("tind", "tdep")) {
    if (time_setting == "tind") {
      # In the time-independent formulas, the time and event columns are named accordingly
      for (event in c("death", "ami", "stroke")) {
        # Create both a spline and categorized instance for updating later
        list_forms[[paste0("form_", event, "_", time_setting, "_spl")]] <- update(base_form, paste0("Surv(time_to_", event, ", ", event, ") ~ . + rcs(time_to_cpap_start)"))
        list_forms[[paste0("form_", event, "_", time_setting, "_cat")]] <- update(base_form, paste0("Surv(time_to_", event, ", ", event, ") ~ . + rcs(time_to_cpap_start)"))
      } 
    } else {
      # In the time-dependent formulas, the time and event columns are start, stop, event always
      list_forms[[paste0("form_", time_setting, "_spl")]] <- update(base_form, "Surv(start, stop, event) ~ .")
      list_forms[[paste0("form_", time_setting, "_cat")]] <- update(base_form, "Surv(start, stop, event) ~ .")
    }
  }
  
  # Add spline or categorized variables, respectively
  for (type in c("spline", "categorized")) {
    i <- 1
    for (form_name in names(list_forms)) {
      if (grepl("_spl", form_name)) {
        list_forms[[i]] <- update(list_forms[[i]], paste(". ~ . + rcs(hba1c) + rcs(GFR) + rcs(kolesterol) + rcs(systoliskt) + rcs(alder):rcs(GFR) + factor(cpap_treated):rcs(GFR)"))
      } else if (grepl("_cat", form_name)) {
        list_forms[[i]] <- update(list_forms[[i]], paste(". ~ . + factor(hba1c_high) + factor(GFR_high) + factor(kolesterol_high) + factor(systoliskt_high) + rcs(alder):factor(GFR_high) + factor(cpap_treated):factor(GFR_high)"))
      }
      i <- i + 1
    } 
  }
  
  # Create a copy that has sleep data for Group 2
  list_forms_sleep <- list_forms
  i <- 1
  for (form in list_forms) {
    list_forms_sleep[[i]] <- update(form, ". ~ . + rcs(IV_AHI) + rcs(IV_ODI) + rcs(IV_AverageSaturation)")
    i <- i + 1
  }
  
  return (list(list_forms, list_forms_sleep))
}
