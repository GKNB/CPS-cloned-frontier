#pragma once

#include <alg/a2a/a2a_fields/field_vectors/W/W_allocpolicies.h>

CPS_START_NAMESPACE

template<typename mf_Policies>
using A2AvectorWunitary_autoAllocPolicies = A2AvectorW_autoAllocPolicies<mf_Policies>;

template<typename mf_Policies>
using A2AvectorWunitary_manualAllocPolicies = A2AvectorW_manualAllocPolicies<mf_Policies>;

CPS_END_NAMESPACE
