#########################################################################
#
#  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
#
#  Copyright (C) 2026 by the ExaDG authors
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program. If not, see <https://www.gnu.org/licenses/>.
#
#########################################################################

# Stand-alone build of the throughput benchmark of the Raviart-Thomas
# momentum operator: only uses the header-only parts of ExaDG and does not
# link to libexadg. Configure with
#
#   cmake -D DEAL_II_DIR=/path/to/deal.II -D CMAKE_BUILD_TYPE=Release \
#     /path/to/exadg/prototypes/throughput_rt_operator

CMAKE_MINIMUM_REQUIRED(VERSION 3.13.4)

FIND_PACKAGE(deal.II 9.7 REQUIRED
  HINTS ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
DEAL_II_INITIALIZE_CACHED_VARIABLES()

IF("${CMAKE_BUILD_TYPE}" STREQUAL "DebugRelease")
  SET(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
ENDIF()

PROJECT(throughput_rt_operator CXX)

ADD_EXECUTABLE(throughput_rt_operator throughput_rt_operator.cc)
TARGET_INCLUDE_DIRECTORIES(throughput_rt_operator PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include
  )
DEAL_II_SETUP_TARGET(throughput_rt_operator)
