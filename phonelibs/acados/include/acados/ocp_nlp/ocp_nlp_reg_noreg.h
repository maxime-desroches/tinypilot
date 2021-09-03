/*
 * Copyright 2019 Gianluca Frison, Dimitris Kouzoupis, Robin Verschueren,
 * Andrea Zanelli, Niels van Duijkeren, Jonathan Frey, Tommaso Sartor,
 * Branimir Novoselnik, Rien Quirynen, Rezart Qelibari, Dang Doan,
 * Jonas Koenemann, Yutao Chen, Tobias Schöls, Jonas Schlagenhauf, Moritz Diehl
 *
 * This file is part of acados.
 *
 * The 2-Clause BSD License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.;
 */


/// \addtogroup ocp_nlp
/// @{
/// \addtogroup ocp_nlp_reg
/// @{

#ifndef ACADOS_OCP_NLP_OCP_NLP_REG_NOREG_H_
#define ACADOS_OCP_NLP_OCP_NLP_REG_NOREG_H_

#ifdef __cplusplus
extern "C" {
#endif



// blasfeo
#include "blasfeo/include/blasfeo_common.h"

// acados
#include "acados/ocp_nlp/ocp_nlp_reg_common.h"



/************************************************
 * dims
 ************************************************/

// use the functions in ocp_nlp_reg_common


/************************************************
 * options
 ************************************************/

typedef struct
{
    int dummy;
} ocp_nlp_reg_noreg_opts;

//
acados_size_t ocp_nlp_reg_noreg_opts_calculate_size(void);
//
void *ocp_nlp_reg_noreg_opts_assign(void *raw_memory);
//
void ocp_nlp_reg_noreg_opts_initialize_default(void *config_, ocp_nlp_reg_dims *dims, void *opts_);
//
void ocp_nlp_reg_noreg_opts_set(void *config_, ocp_nlp_reg_dims *dims, void *opts_, char *field, void* value);



/************************************************
 * memory
 ************************************************/

typedef struct
{
    int dummy;
} ocp_nlp_reg_noreg_memory;

//
acados_size_t ocp_nlp_reg_noreg_memory_calculate_size(void *config, ocp_nlp_reg_dims *dims, void *opts);
//
void *ocp_nlp_reg_noreg_memory_assign(void *config, ocp_nlp_reg_dims *dims, void *opts, void *raw_memory);

/************************************************
 * workspace
 ************************************************/

// not needed

/************************************************
 * functions
 ************************************************/

//
void ocp_nlp_reg_noreg_config_initialize_default(ocp_nlp_reg_config *config);



#ifdef __cplusplus
}
#endif

#endif  // ACADOS_OCP_NLP_OCP_NLP_REG_NOREG_H_

/// @}
/// @}
