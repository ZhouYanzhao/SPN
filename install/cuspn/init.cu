#include "luaT.h"
#include "THC.h"
#include "utils.h"

#include "SoftProposal.cu"

LUA_EXTERNC DLL_EXPORT int luaopen_libcuspn(lua_State *L);

int luaopen_libcuspn(lua_State *L)
{
  lua_newtable(L);

  cuspn_SP_init(L);
  
  return 1;
}
