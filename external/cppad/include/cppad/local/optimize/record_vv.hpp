// $Id$
# ifndef CPPAD_LOCAL_OPTIMIZE_RECORD_VV_HPP
# define CPPAD_LOCAL_OPTIMIZE_RECORD_VV_HPP
/* --------------------------------------------------------------------------
CppAD: C++ Algorithmic Differentiation: Copyright (C) 2003-16 Bradley M. Bell

CppAD is distributed under multiple licenses. This distribution is under
the terms of the
                    Eclipse Public License Version 1.0.

A copy of this license is included in the COPYING file of this distribution.
Please visit http://www.coin-or.org/CppAD/ for information on other licenses.
-------------------------------------------------------------------------- */
/*!
\file record_vv.hpp
Record an operation of the form (variable op variable).
*/
// BEGIN_CPPAD_LOCAL_OPTIMIZE_NAMESPACE
namespace CppAD { namespace local { namespace optimize  {
/*!
Record an operation of the form (variable op variable).

\param var2op
mapping from old variable index to old operator index.

\param op_info
mapping from old index to operator index to operator information

\param old2new
mapping from old operator index to information about the new recording.

\param current
is the index in the old operation sequence for
the variable corresponding to the result for the current operator.
We use the notation i_op = var2op[current].
It follows that  NumRes( op_info[i_op].op ) > 0.
If 0 < j_op < i_op, either op_info[j_op].csum_connected,
op_info[j_op].usage = 0, or old2new[j_op].new_var != 0.

\param npar
is the number of parameters corresponding to the old operation sequence.

\param par
is a vector of length npar containing the parameters
the old operation sequence; i.e.,
given a parameter index i < npar, the corresponding parameter value is par[i].

\param rec
is the object that will record the new operations.

\return
is the operator and variable indices in the new operation sequence.

\param op
is the operator that we are recording which must be one of the following:
AddvvOp, DivvvOp, MulvvOp, PowvvOp, SubvvOp, ZmulvvOp.

\param arg
is the vector of arguments for this operator.
*/
template <class Base>
struct_size_pair record_vv(
	const vector<addr_t>&                              var2op         ,
	const vector<struct_op_info>&                      op_info        ,
	const CppAD::vector<struct struct_old2new>&        old2new        ,
	size_t                                             current        ,
	size_t                                             npar           ,
	const Base*                                        par            ,
	recorder<Base>*                                    rec            ,
	OpCode                                             op             ,
	const addr_t*                                      arg            )
{
# ifndef NDEBUG
	switch(op)
	{	case AddvvOp:
		case DivvvOp:
		case MulvvOp:
		case PowvvOp:
		case SubvvOp:
		case ZmulvvOp:
		break;

		default:
		CPPAD_ASSERT_UNKNOWN(false);
	}
# endif
	CPPAD_ASSERT_UNKNOWN( size_t(arg[0]) < current );
	CPPAD_ASSERT_UNKNOWN( size_t(arg[1]) < current );
	addr_t new_arg[2];
	new_arg[0]   = old2new[ var2op[arg[0]] ].new_var;
	new_arg[1]   = old2new[ var2op[arg[1]] ].new_var;
	rec->PutArg( new_arg[0], new_arg[1] );

	struct_size_pair ret;
	ret.i_op  = rec->num_op_rec();
	ret.i_var = rec->PutOp(op);
	CPPAD_ASSERT_UNKNOWN( 0 < new_arg[0] && size_t(new_arg[0]) < ret.i_var );
	CPPAD_ASSERT_UNKNOWN( 0 < new_arg[1] && size_t(new_arg[1]) < ret.i_var );
	return ret;
}

} } } // END_CPPAD_LOCAL_OPTIMIZE_NAMESPACE


# endif
