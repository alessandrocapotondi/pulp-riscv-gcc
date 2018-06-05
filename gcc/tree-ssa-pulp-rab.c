/* PULP RAB management pass. L3 access SW instrumentation for the GNU compiler.

   Copyright (C) 2002-2015 Free Software Foundation, Inc.
   Contributed by Andrea Marongiu <a.marongiu@iis.ee.ethz.ch>

This file is part of GCC.

GCC is free software; you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation; either version 3, or (at your option) any
later version.

GCC is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
for more details.

You should have received a copy of the GNU General Public License
along with GCC; see the file COPYING3.  If not see
<http://www.gnu.org/licenses/>.  */


/* This pass implements protection of memory references meant to reach L3 (DRAM)
 * memory in a heterogeneous, PULP-based system featuring a Remap Address Block (RAB).
 * We want to be able to share pointers between the ARM-based host and PULP, so since
 * the ARM operates in virtual memory the RAB takes care of doing the virtual-to-physical
 * translation. However, if a mapping for a particular virtual address (i.e., the page
 * this address belongs to) is not present, the HW is not capable of stalling the core
 * that issued the LD/ST while the page miss is managed, which leads to a crash.
 * 
 * The approach proposed here assumes that the SW takes care of protecting every LD/ST
 * to DRAM by emitting a TEST_READ instruction (i.e., a call to the runtime function
 * "GOMP_pulp_rab_tryread") before such memory accesses. The function checks that the
 * page is in the RAB, and if this is not the case it puts the calling core into sleep and 
 * initiates the miss handling routine. Once this is finished a wakeup signal is sent to
 * the core.
 * 
 * The instrumentation process starts by discriminating the potentially dangerous accesses
 * based on the content of the .omp_data_i metadata generated during OpenMP expansion for 
 * the PRAGMA OMP TARGET directive.
 * Let us consider the following pointer chasing sample code:
 * 
 *   #pragma omp target map(from:head) map(from:num_elements)
 *   {
 *     int i;
 *     
 *     for (i=0; i<num_elements; i++)
 *     {
 *       head = head->next;
 *     }
 *   }
 * 
 * which gets translated in the following code
 * 
 * 1   _app_main._omp_fn.0 (struct .omp_data_t.8 & restrict .omp_data_i)
 * 2   {
 * 3     int i;
 * 4     int * _6;
 * 5     int num_elements.7_7;
 * 6     struct node * * _8;
 * 7     struct node * head.5_9;
 * 8     unsigned int * head.6_10;
 *
 * 9     <bb 2>:
 *
 * 10    <bb 3>:
 * 11    # i_1 = PHI <0(2), i_13(5)>
 * 12    _6 = *.omp_data_i_5(D).num_elements;
 * 13    num_elements.7_7 = *_6;
 * 14    if (i_1 < num_elements.7_7)
 * 15      goto <bb 5>;
 * 16    else
 * 17      goto <bb 4>;
 *
 * 18    <bb 4>:
 * 19    return;
 *
 * 20    <bb 5>:
 * 21    _8 = *.omp_data_i_5(D).head;
 * 22    head.5_9 = *_8;
 * 23    head.6_10 = head.5_9->next;
 * 24    *_8 = head.6_10;
 * 25    i_13 = i_1 + 1;
 * 26    goto <bb 3>;
 * 27  }
 * 
 * 
 * we first look for those statements where the base address of the variables that
 * reside in L3 are read. This happens on lines 12 and 21, for the variables
 * <num_elements> and <head>, which where used in MAP clauses, and thus expanded as
 * fields of the .omp_data_i metadata.
 * The address of such variables is read into temporaries (_6 and _8, respectively),
 * and our job is to ensure that the content of such temporaries is instrumented with
 * a RAB lookup before we can safely access the address.
 * 
 * After that, we need to propoagate the necessity of instrumenting accesses to all 
 * the statements (and operands therein) for which the temporary variable at the lhs 
 * is a reaching definition.
 * 
 * Note that in the general case we need to be very conservative and repeat the lookup
 * before every occurrence of the "dangerous" LD/ST, since we can make no assumptions
 * regarding the presence of the page in the RAB (an eviction may have occurred since
 * the last time we handled the miss).
 * 
 * If the programmer says so, we can only do the instrumentation once per base address
 * and collect all the TRYREAD instructions at the beginning of the block (this has a
 * positive effect on page miss management time).
 * 
 */

#include "config.h"
#include "system.h"
#include "coretypes.h"
#include "backend.h"
#include "rtl.h"
#include "tree.h"
#include "gimple.h"
#include "cfghooks.h"
#include "tree-pass.h"
#include "ssa.h"
#include "gimple-pretty-print.h"
#include "fold-const.h"
#include "calls.h"
#include "cfganal.h"
#include "tree-eh.h"
#include "gimplify.h"
#include "gimple-iterator.h"
#include "tree-cfg.h"
#include "tree-ssa-loop-niter.h"
#include "tree-into-ssa.h"
#include "tree-dfa.h"
#include "cfgloop.h"
#include "tree-scalar-evolution.h"
#include "tree-chkp.h"
#include "tree-ssa-propagate.h"
#include "gimple-fold.h"

#if 0
#include "config.h"
#include "system.h"
#include "coretypes.h"
#include "tm.h"
#include "hash-set.h"
#include "machmode.h"
#include "vec.h"
#include "double-int.h"
#include "input.h"
#include "alias.h"
#include "symtab.h"
#include "wide-int.h"
#include "inchash.h"
#include "tree.h"
#include "fold-const.h"
#include "calls.h"
#include "gimple-pretty-print.h"
#include "predict.h"
#include "hard-reg-set.h"
#include "function.h"
#include "dominance.h"
#include "cfg.h"
#include "cfganal.h"
#include "basic-block.h"
#include "tree-ssa-alias.h"
#include "internal-fn.h"
#include "tree-eh.h"
#include "gimple-expr.h"
#include "is-a.h"
#include "gimple.h"
#include "gimplify.h"
#include "gimple-iterator.h"
#include "gimple-ssa.h"
#include "tree-cfg.h"
#include "tree-phinodes.h"
#include "ssa-iterators.h"
#include "stringpool.h"
#include "tree-ssanames.h"
#include "tree-ssa-loop-niter.h"
#include "tree-into-ssa.h"
#include "hashtab.h"
#include "rtl.h"
#include "flags.h"
#include "statistics.h"
#include "real.h"
#include "fixed-value.h"
#include "insn-config.h"
#include "expmed.h"
#include "dojump.h"
#include "explow.h"
#include "emit-rtl.h"
#include "varasm.h"
#include "stmt.h"
#include "expr.h"
#include "tree-dfa.h"
#include "tree-pass.h"
#include "cfgloop.h"
#include "tree-scalar-evolution.h"
#include "tree-chkp.h"
#include "tree-ssa-propagate.h"
#include "gimple-fold.h"
#endif

#include "gimple-walk.h"
#include "tree-cfgcleanup.h"


#define PULP_NEEDS_RAB_INSTRUMENTATION_FLAG(NODE)	(TYPE_TO_INSTRUMENT(NODE))
#define PULP_NEEDS_RAB_INSTRUMENTATION_P(NODE)		(TYPE_TO_INSTRUMENT(NODE) == 1)
#define PULP_SET_NEEDS_RAB_INSTRUMENTATION(NODE)	(TYPE_TO_INSTRUMENT(NODE) = 1)

#define PULP_NEEDS_RAB_INSTRUMENTATION_STMT_P(NODE)	gimple_needs_rab_instrumentation_stmt(NODE)
#define PULP_SET_NEEDS_RAB_INSTRUMENTATION_STMT(NODE)	gimple_set_needs_rab_instrumentation_stmt(NODE,1)

#define PULP_MARKED_BY_SSA_RAB_PASS_STMT_P(NODE)	gimple_marked_by_ssa_rab_pass_stmt(NODE)
#define PULP_SET_MARKED_BY_SSA_RAB_PASS_STMT(NODE,FLAG)	gimple_set_marked_by_ssa_rab_pass_stmt(NODE,FLAG)



bool conservative;
static sbitmap processed;


/* Called via walk_tree, look for uses of SSA_VAR in a STMT  */

static tree
find_ssa_name_in_expr (tree *tp, int *walk_subtrees, void *data)
{
  struct walk_stmt_info *wi = (struct walk_stmt_info *) data;
  tree *ssa_var = (tree *) wi->info;

  if (TREE_CODE (*tp) == SSA_NAME)
    {
      if (*tp == *ssa_var)
	return *tp;

      *walk_subtrees = 0;
    }

  return NULL_TREE;
}


/* Returns true if EXPR contains SSA_REF */
static bool
expr_contains_ref_p (tree expr, tree ssa_ref)
{
  struct walk_stmt_info wi;
  memset (&wi, 0, sizeof (wi));
  wi.info = &ssa_ref;

  return (walk_tree (&expr, find_ssa_name_in_expr, &wi, NULL));
}

/* Returns true if EXPR is a simple dereference of SSA_REF */
static bool
is_simple_dereference_p (tree expr, tree ssa_ref)
{
  return (TREE_CODE (expr) == MEM_REF && TREE_OPERAND (expr, 0) == ssa_ref);
}

#ifndef BLOCKING_TRYREAD

static tree
build_fndecl (const char *name, tree type)
//build_fndecl (const char *name, tree type, tree attrs)
{
  tree   id = get_identifier (name);
  tree decl = build_decl (BUILTINS_LOCATION, FUNCTION_DECL, id, type);

  TREE_PUBLIC (decl)         = 1;
  DECL_EXTERNAL (decl)       = 1;

//   DECL_BUILT_IN_CLASS (decl) = cl;

//   DECL_FUNCTION_CODE (decl)  = (enum built_in_function) function_code;

  /* DECL_FUNCTION_CODE is a bitfield; verify that the value fits.  */
//   gcc_assert (DECL_FUNCTION_CODE (decl) == function_code);

//   if (library_name)
//     {
//       tree libname = get_identifier (library_name);
//       SET_DECL_ASSEMBLER_NAME (decl, libname);
//     }

  /* Possibly apply some default attributes to this built-in function.  */
//   if (attrs)
//     decl_attributes (&decl, attrs, ATTR_FLAG_BUILT_IN);
//   else
//     decl_attributes (&decl, NULL_TREE, 0);

  set_call_expr_flags (decl, ECF_NOTHROW | ECF_LEAF);
  return decl;
}

/* Helper function for update_gimple_call and update_call_from_tree.
   A GIMPLE_CALL STMT is being replaced with GIMPLE_CALL NEW_STMT.  */

static void
finish_update_gimple_call (gimple_stmt_iterator *si_p, gimple *new_stmt,
         gimple *stmt)
{
//   gimple_call_set_lhs (new_stmt, gimple_call_lhs (stmt));
  move_ssa_defining_stmt_for_defs (new_stmt, stmt);
  gimple_set_vuse (new_stmt, gimple_vuse (stmt));
  gimple_set_vdef (new_stmt, gimple_vdef (stmt));
  gimple_set_location (new_stmt, gimple_location (stmt));
  if (gimple_block (new_stmt) == NULL_TREE)
    gimple_set_block (new_stmt, gimple_block (stmt));
  gsi_replace (si_p, new_stmt, false);
}

static void
emit_trywrite_call (tree ref, gimple *stmt)
{
//   tree rhs = gimple_assign_rhs1 (stmt);
// //   gimple call = gimple_build_call (builtin_decl_explicit (BUILT_IN_GOMP_PULP_RAB_TRYWRITE),
// // 				   2, ref, rhs);
// 
// /* Andrea - 19/5/17 - Using pointer types for function parameters disables LTO inlining of
//    TRYREAD and TRYWRITE. We use unsigned integer instead, with appropriate casts in the
//    implementation of the functions.
// */
// //  tree ftype = build_function_type_list (ptr_type_node, ptr_type_node, 
// //					 integer_type_node, NULL_TREE);
//   tree ftype = build_function_type_list (void_type_node, unsigned_type_node, 
// 					 unsigned_type_node, NULL_TREE);
// 
//   gimple call = gimple_build_call (build_fndecl ("GOMP_pulp_RAB_trywrite", ftype),
// 				   2, ref, rhs);
//   gimple_stmt_iterator gsi = gsi_for_stmt (stmt);
//   finish_update_gimple_call (&gsi, call, stmt);

/****************************************************************/

  tree rhs = gimple_assign_rhs1 (stmt);
  tree stmt_write_type = TREE_TYPE (rhs);
  
  const char *fname = (TREE_CODE (stmt_write_type) == INTEGER_TYPE) ?
		       "GOMP_pulp_RAB_trywrite" : "GOMP_pulp_RAB_trywrite_f";
		       
  tree ftype = build_function_type_list (void_type_node, unsigned_type_node, 
					 stmt_write_type, NULL_TREE);

  gimple *call = gimple_build_call (build_fndecl (fname, ftype),
				   2, ref, rhs);
  
  gimple_stmt_iterator gsi = gsi_for_stmt (stmt);
  finish_update_gimple_call (&gsi, call, stmt);
}

#endif	/* BLOCKING_TRYREAD */


static void
emit_tryread_call (tree ref, gimple *stmt)
{
#ifndef BLOCKING_TRYREAD

/* Andrea - 19/5/17 - Using pointer types for function parameters disables LTO inlining of
   TRYREAD and TRYWRITE. We use unsigned integer instead, with appropriate casts in the
   implementation of the functions.
*/
  
/* NOTE: (6/7/17) TRYREAD returns an UNSIGNED INT. When assigning the result to a 
 * different type we need to handle the cast
 */
// NOTE: OLD VERSION (6/7/17)
//   tree rhs = gimple_assign_rhs1 (stmt);
//   tree ftype = build_function_type_list (/*TREE_TYPE (rhs)*/unsigned_type_node, unsigned_type_node, 
// 					 NULL_TREE);
// 
//   gimple call = gimple_build_call (build_fndecl ("GOMP_pulp_RAB_tryread", ftype),
// 				   1, ref);//fold_convert (unsigned_type_node, ref));
//   
//   tree lhs = make_ssa_name (TREE_TYPE (rhs));
//   gimple_call_set_lhs (call, lhs);
//   gimple_stmt_iterator gsi = gsi_for_stmt (stmt);
//   
//   gsi_insert_before (&gsi, call, GSI_NEW_STMT);
//   
//   /* Fix the RHS of the original statement */
//   gimple_assign_set_rhs1 (stmt, lhs);
// 
//   update_stmt (stmt);

/***********************************************************************/

  /* Detect the type read from the original statement  */
//   tree stmt_read_type = TREE_TYPE (gimple_assign_rhs1 (stmt));
// 
//   const char *fname = "GOMP_pulp_RAB_tryread";
// 
//   tree ftype = build_function_type_list (build_pointer_type (unsigned_type_node), unsigned_type_node, 
// 					 NULL_TREE);
//   gimple call = gimple_build_call (build_fndecl (fname, ftype),
// 				   1, ref);//fold_convert (unsigned_type_node, ref));
// 
//   tree lhs = make_ssa_name (unsigned_type_node);
//   gimple_call_set_lhs (call, lhs);
//   
//   /* Now build the cast expression */
//   
//   tree addr_lhs = make_ssa_name (build_pointer_type (unsigned_type_node));
//   gimple addr_stmt = gimple_build_assign (addr_lhs, build_fold_addr_expr (lhs));
//   
// //   tree conv_lhs = make_ssa_name (build_pointer_type (stmt_read_type));
// //   gimple conv_stmt = gimple_build_assign (conv_lhs, fold_convert (build_pointer_type (stmt_read_type), 
// // 							  addr_lhs));
//   
// //   tree read_lhs = make_ssa_name (stmt_read_type);
// //   gimple read_stmt = gimple_build_assign (read_lhs, build_simple_mem_ref (conv_lhs));
//   
//   tree read_lhs = build_simple_mem_ref (addr_lhs);//conv_lhs);
//   
//   gimple_stmt_iterator gsi = gsi_for_stmt (stmt);
//   
//   /* Insert the TRYREAD call and cast (reverse order) */
// //   gsi_insert_before (&gsi, read_stmt, GSI_NEW_STMT);
// //   gsi_insert_before (&gsi, conv_stmt, GSI_NEW_STMT);
//   gsi_insert_before (&gsi, addr_stmt, GSI_NEW_STMT);
//   gsi_insert_before (&gsi, call, GSI_NEW_STMT);
//   
//   /* Fix the RHS of the original statement */
//   gimple_assign_set_rhs1 (stmt, read_lhs);
  
/*****************************************************************/

  tree stmt_read_type = TREE_TYPE (gimple_assign_rhs1 (stmt));

  const char *fname = (TREE_CODE (stmt_read_type) == INTEGER_TYPE) ?
		       "GOMP_pulp_RAB_tryread" : "GOMP_pulp_RAB_tryread_f";

  tree ftype = build_function_type_list (stmt_read_type, unsigned_type_node, 
					 NULL_TREE);
  gimple *call = gimple_build_call (build_fndecl (fname, ftype),
				   1, ref);//fold_convert (unsigned_type_node, ref));
				   
  tree lhs = make_ssa_name (stmt_read_type);
  gimple_call_set_lhs (call, lhs);
  
  gimple_stmt_iterator gsi = gsi_for_stmt (stmt);
  gsi_insert_before (&gsi, call, GSI_NEW_STMT);
  
  /* Fix the RHS of the original statement */
  gimple_assign_set_rhs1 (stmt, lhs);
  
  update_stmt (stmt);
  
#else	/*BLOCKING_TRYREAD */

  gimple *call = gimple_build_call (builtin_decl_explicit (BUILT_IN_GOMP_PULP_RAB_TRY_READ),
				   1, ref);
  gimple_stmt_iterator gsi = gsi_for_stmt (stmt);
//   gsi_insert_after (&gsi, call, GSI_CONTINUE_LINKING);
  gsi_insert_before (&gsi, call, GSI_SAME_STMT);
  
#endif	/* BLOCKING_TRYREAD */
}

/* This function checks the use of REF within the current STMT, and establishes whether
 * this use simply accesses the original MEM_REF, or builds a new expression based on that.
 * In the latter case we must propagate the analysis to the LHS of the new STMT.
 */
static bool
should_propagate_analysis (tree ref)
{
  return (TREE_CODE (TREE_TYPE(ref)) != POINTER_TYPE) ? false : true;
}
#if 0
should_propagate_analysis (tree ssa_var, tree ref, gimple stmt)
{
  /* This is a "HACK" for COMPONENT_REF expressions like struct->field, which escape
     instrumentation. Here we build their address expression and instrument it */
  if (TREE_CODE (TREE_TYPE(ref)) != POINTER_TYPE)// && 
//       TREE_CODE (ref) != COMPONENT_REF)
    return false;
  
  if (TREE_CODE (ref) != MEM_REF)
    /* Only MEM_REFs need to be instrumented, but if the current REF
     * is part of a USE-DEF chain that might eventually expose a MEM_REF
     * we need to continue the propagation process
     */
    return true;
  
  /* The REF was an address. Instrumentation done! */
  return true;
}
#endif

/* Visit all STMTs that contain a use of SSA_VAR.
 * In the process check is the address contained in SSA_VAR was dereferenced (READ/WRITE).
 * Instead than returning a BOOL (address was dereferenced) we return a TREE, to deal 
 * with the case when data structures are dereferenced (i.e., one of their fields is accessed.
 * For example:
 * 
 * tmp = struct->field
 * 
 * would become
 * 
 * 1) _1 = struct;
 * 2) tmp = _1->field;
 * 
 * If we returned TRUE when analyzing statement 2) (SSA_VAR _1 was dereferenced) the pass
 * would instrument the access to the data structure base address
 * 
 * 1) _1 = struct;
 * 2) __builtin_omp_tryrab_read (_1);
 * 3) tmp = _1->field;
 * 
 * but we're actually accessing the data structure with an offset (the field). So we should
 * rather be doing the following:
 * 
 * 1) _1 = struct;
 * 2) _2 = &_1->field;
 * 3) __builtin_omp_tryrab_read (_2);
 * 4) tmp = _1->field;
 * 
 * For this reason this function returns a TREE, which contains the expression that accesses
 * the SSA_VAR (i.e., the correct expression to be instrumented).
 */
static void
propagate_instrumentation (tree ssa_var, bool emit_tryx)
{
//  use_operand_p use_p;
  imm_use_iterator iter;
  gimple *stmt;
  tree ref;
  bool is_trywrite = false;

  FOR_EACH_IMM_USE_STMT (stmt, iter, ssa_var)
  {
    /*TODO: Now the algorithm should do the following steps:
      * 1) Distinguish between plain uses of the offloaded var reference and 
      * dereferenced accesses. A plain use implies reading (rhs) or writing (lhs)
      * the pointer to the offloaded var. Clearly the second does not make sense
      * (we would be overriding the pointer itself), so plain uses should always 
      * be on the RHS. In the most conservative case we need to always instrument
      * a read-access to this pointer, because is possible that even if we already
      * have a mapping in the rab for the pointers to the offloaded vars (which 
      * could be done once for always above) the mapping has been evicted due to
      * intervening RAB misses. Thus at this stage we check:
      * a) that plain uses only happen on the RHS
      * b) where dereferenced accesses happen
      * 
      * 2) if a dereferenced access happens on the LHS this means we are writing
      * to the offloaded var. This access needs to be instrumented, but it cannot 
      * further escape, so we are done.
      * 
      * 3) if a dereferenced access happens on the RHS this means we are reading
      * the location of memory pointed by the offload variable. In this case we need
      * to further check the pointed type. If this is a TYPE_POINTER itself we need
      * to further propagate the reaching definitions analysis, as we may recursively
      * be accessing main memory (pointer chasing) */
    
    tree lhs = gimple_assign_lhs (stmt);
    tree rhs = gimple_assign_rhs1 (stmt);

#define USE_IS_ON_LHS	expr_contains_ref_p (lhs, ssa_var)    
#define USE_IS_ON_RHS	expr_contains_ref_p (rhs, ssa_var)

    /* First of all we check if the use is on the RHS, because in this case we
     * might be simply assigning its dereferenced value to a SSA temporary.
     * In this case we don't need to instrument the use on the RHS itself, but 
     * we can rather propagate the necessity for the instrumentation on the LHS
     */ 
   if (USE_IS_ON_RHS)
    { 
////      if (should_propagate_analysis (ssa_var, rhs, stmt))
     if (should_propagate_analysis (rhs))
      {
	/* The OFFLOADED_VAR reference is used on the RHS, and is a pointer.
	 * If STMT is of the type: SSA_VAR = ADDRESS - func (OFFLOADED_VAR)
	 * this means we are reading the address into a temporary for later
	 * use. In this case we need to propagate the instrumentation to
	 * reaching defs/uses of its LHS */
// 	gcc_assert (TREE_CODE (lhs) == SSA_NAME);
	PULP_SET_NEEDS_RAB_INSTRUMENTATION_STMT (stmt);

        /* This statement was marked for instrumentation by this pass,
	 * not OMP expansion. Should emit tryx instructions for this
	 * statement and those derived from it.
         */
	PULP_SET_MARKED_BY_SSA_RAB_PASS_STMT (stmt, true);
      }
    }
 
    is_trywrite = USE_IS_ON_LHS;
    ref = is_trywrite ? lhs : rhs;  
    
    // Don't emit tryx instructions for statements that read into omp_data_i
    // as the addresses belong to a copy held in the accelerator memory
    // (i.e., they don't reach main memory, so no need for RAB instrumentation).
    if (TREE_CODE (ref) != ARRAY_REF && !emit_tryx)
	continue;


    /* Now let's determine if the use implies a dereference of the original address */
    if (is_simple_dereference_p (ref, ssa_var)	||
	//TREE_CODE (rhs) == MEM_REF
        TREE_CODE (ref) == ARRAY_REF
       )
    {
      if (is_trywrite)
	emit_trywrite_call (build_fold_addr_expr (ref), stmt);
      else
	emit_tryread_call (build_fold_addr_expr (ref), stmt);
    }
    
    if(TREE_CODE (ref) == COMPONENT_REF)
    {
      /* EMIT tryread call here and now! */
      if (is_trywrite)
	emit_trywrite_call (build_fold_addr_expr (ref), stmt);
      else
	emit_tryread_call (build_fold_addr_expr (ref), stmt);
    }
    
    if (USE_IS_ON_LHS)
    {
      /* This is a write. Not used today, but we'd better keep the discrimination for the future */
    }
  }
}

#define HOST_POINTERS_ARE_COPIED_TO_PULP_MEMORY

static void
pulp_instrument_rab_accesses ()
{
  basic_block bb;
  gimple_stmt_iterator gsi;
  gimple *stmt;

  FOR_EACH_BB_FN (bb, cfun)
  {
      /* FIXME: (maybe) First of all we need to mark all statements with NEED_INSTRUMENTATION_SET
     * as <marked during the OpenMP expansion pass>. Marking them back then doesn't
     * work, as we use gimple(stmt)->plf, which is not propagated correctly across
     * the passes (information might have been lost at this stage).
     */
    for (gsi = gsi_start_bb (bb); !gsi_end_p (gsi); gsi_next (&gsi))
    {
      stmt = gsi_stmt (gsi);

//       if (gimple_code (stmt) != GIMPLE_ASSIGN)
// 	continue;
//       
//       if (!PULP_NEEDS_RAB_INSTRUMENTATION_STMT_P (stmt))
// 	continue;

#ifdef HOST_POINTERS_ARE_COPIED_TO_PULP_MEMORY
      /* During expansion of #pragma omp target, host pointers are
       * marshalled into metadata as the addresses of the variables 
       * that host such pointers.
       * Example:
       * 
       * int b[10];
       * 
       * omp_data_arr.b = &b;
       * 
       * that is later made visible to the  accelerator (offloaded 
       * function parameter) as omp_data_i.
       * If the runtime implementation of the GOMP_target passes such
       * pointers on a as-is basis, then this will imply protection of
       * accesses to such variables (tryX instrumentation) when we want
       * to construct the address of the array element to be accesses
       * from DRAM. Concretely;
       * 
       * int tmp = b[5];
       * 
       * which is by default translated into 
       * 
       * int *t1 = omp_data_i->b + 5 * sizeof (int);
       * int tmp = *t1;
       * 
       * becomes
       * 
       * unsigned int t1 = tryread (omp_data_i->b);
       * int *t2 = t1 + 5 * sizeof (int);
       * int tmp = tryread (t2);
       * 
       * If the GOMP_target copies such pointers (i.e., the addresses
       * they point to) into local copies of the omp_data_i structure
       * that reside in some PULP internal memory, then we don't need
       * to do such instrumentation. The example above becomes
       * 
       * int *t1 = omp_data_i->b + 5 * sizeof (int);
       * int tmp = tryread (t1);
       * 
       * with less instrumentation operations.
       */
      PULP_SET_MARKED_BY_SSA_RAB_PASS_STMT (stmt, false);
#else
      PULP_SET_MARKED_BY_SSA_RAB_PASS_STMT (stmt, true);
#endif
    }
  }
  
  FOR_EACH_BB_FN (bb, cfun)
  {
    /* Check all statements in the block.  */
    for (gsi = gsi_start_bb (bb); !gsi_end_p (gsi); gsi_next (&gsi))
    {
      stmt = gsi_stmt (gsi);

      /* Only assignment statement can be marked for instrumentation */
      if (gimple_code (stmt) != GIMPLE_ASSIGN)
	continue;

      /* If the LHS is not an SSA_NAME can't be an interesting statement to us */
      tree lhs = gimple_op (stmt, 0);
      if (TREE_CODE (lhs) != SSA_NAME)
	continue;
      
      /* If the LHS of this assign statement is an SSA_NAME, we need to check
	 if this particular data ref actually needs instrumentation.
	 This was either marked during OpenMP lowering, or it is
	 marked during statement traversal in this pass.
      */
      if (!PULP_NEEDS_RAB_INSTRUMENTATION_STMT_P (stmt))
	continue;
      
      /* We have found an assign statement that reads out of
         the metadata the address of the offloaded var, or that
         uses this address to point to other memory locations that
         might need instrumentation.
         We need follow the use-def chain (propagate_instrumentation)
         to ensure that all accesses to DRAM are instrumented.
         propagate_instrumentation walks all statements that use the
         LHS of the current STMT, and marks those that may "escape".
         The function returns TRUE if one of the walked statemenst has
         accessed (read/write) LHS.
      */
      propagate_instrumentation (lhs, PULP_MARKED_BY_SSA_RAB_PASS_STMT_P (stmt));
    }
  }
}



static unsigned int
perform_rab_instrumentation ()
{

#define fname IDENTIFIER_POINTER(DECL_NAME(cfun->decl))

  
  /* We only need to instrument functions generated by the
   * expansion of OpenMP directives TARGET, PARALLEL and TASK */
  if (!DECL_ARTIFICIAL (cfun->decl) || !strstr (fname, "omp_fn"))
    /* and DECLARE TARGET... */
    if (!lookup_attribute ("omp declare target", DECL_ATTRIBUTES (cfun->decl)))
      return 1;

  conservative = true;
  if (flag_safe_single_rab_instrumentation)
    conservative = false;

  calculate_dominance_info (CDI_DOMINATORS);

  processed = sbitmap_alloc (num_ssa_names + 1);
  bitmap_clear (processed);
  pulp_instrument_rab_accesses ();
  sbitmap_free (processed);

  cleanup_tree_cfg ();
  free_dominance_info (CDI_DOMINATORS);
  
  update_ssa (TODO_update_ssa);
  
  return 0;
}


/* Pass entry point. */

static unsigned int
tree_ssa_pulp_rab (void)
{
  return perform_rab_instrumentation ();
}


namespace {

const pass_data pass_data_pulp_rab =
{
  GIMPLE_PASS, /* type */
  "pulp_rab", /* name */
  OPTGROUP_NONE, /* optinfo_flags */
  TV_PULP_RAB, /* tv_id */
  ( PROP_cfg | PROP_ssa ), /* properties_required */
  0, /* properties_provided */
  0, /* properties_destroyed */
  0, /* todo_flags_start */
  0, /* todo_flags_finish */
};

class pass_pulp_rab : public gimple_opt_pass
{
public:
  pass_pulp_rab (gcc::context *ctxt)
    : gimple_opt_pass (pass_data_pulp_rab, ctxt)
  {}

  /* opt_pass methods: */
  opt_pass * clone () { return new pass_pulp_rab (m_ctxt); }
  virtual bool gate (function *) { return flag_pulp_rab != 0; }
  virtual unsigned int execute (function *) { return tree_ssa_pulp_rab (); }

}; // class pass_pulp_rab

} // anon namespace

gimple_opt_pass *
make_pass_pulp_rab (gcc::context *ctxt)
{
  return new pass_pulp_rab (ctxt);
}
