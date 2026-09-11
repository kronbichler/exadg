/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2025 by Martin Kronbichler, Shubham Goswami,
 *  Richard Schussnig
 *
 *  This file is dual-licensed under the Apache-2.0 with LLVM Exception (see
 *  https://spdx.org/licenses/Apache-2.0.html and
 *  https://spdx.org/licenses/LLVM-exception.html) and the GNU General Public
 *  License as published by the Free Software Foundation, either version 3 of
 *  the License, or (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License in the top-level LICENSE file for
 *  more details.
 *  ______________________________________________________________________
 */

#ifndef EXADG_UTILITIES_FILENAME_UTILITIES_H_
#define EXADG_UTILITIES_FILENAME_UTILITIES_H_

// C/C++
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <string>

namespace ExaDG
{
/**
 * Converts a floating-point value to a filename-safe string in scientific
 * notation (e.g., -1.25481651651684e-05 -> "m1c25481651651684em05"), replacing
 * the characters '.', '-', and '+' (which are not allowed or not
 * distinguishable in all file systems) so that distinct, closely-spaced values
 * are guaranteed to map to distinct filenames.
 */
inline std::string
value_to_filename_string(double const value, unsigned int const n_digits)
{
  std::ostringstream oss;
  oss << std::scientific << std::setprecision(n_digits) << value;
  std::string result = oss.str();

  std::replace(result.begin(), result.end(), '.', 'c');
  std::replace(result.begin(), result.end(), '-', 'm');
  std::replace(result.begin(), result.end(), '+', 'p');

  return result;
}

} // namespace ExaDG

#endif /* EXADG_UTILITIES_FILENAME_UTILITIES_H_ */
