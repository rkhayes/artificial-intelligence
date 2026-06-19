"""
MNIST Label Parser (Manual Implementation)

Author: Diogo Bonofre (diogobonofre)

Description:
    This script manually parses the binary structure of the MNIST training labels file
    (train-labels-idx1-ubyte) without using high-level data loading libraries.

    It serves as an exercise in understanding:
    1. Binary File I/O: Reading raw bytes ('rb' mode) vs text.
    2. Endianness: Converting Big-Endian (High-byte first) data from the file
       to the system's native integer format (usually Little-Endian on Intel/AMD).
    3. Metadata Parsing: extracting the "Magic Number" and "Item Count" to validate
       the file structure before reading the payload.
    4. Data Validation: Ensuring all extracted labels fall within the 0-9 range.

File Format (IDX1-UBYTE):
    [Offset] [Type]          [Value]          [Description]
    0000     32-bit Integer  2049             Magic Number (0x00000801)
    0004     32-bit Integer  60000            Number of Items
    0008     Unsigned Byte   ??               Label 1
    0009     Unsigned Byte   ??               Label 2
    ...      ...             ...              ...

Reference:
    - MNIST Database: http://yann.lecun.org/exdb/mnist/
    - Binary File Parsing: https://youtu.be/oSCVb4Ts-G4
    - IDX File Format: https://www.fon.hum.uva.nl/praat/manual/IDX_file_format.html
"""

import struct
from sys import byteorder

FILE_PATHS = [
    "./data/mnist/train-labels.idx1-ubyte",
    "./data/mnist/t10k-labels.idx1-ubyte",
]


def main():
    print(f"System Byteorder: {byteorder}")

    for path in FILE_PATHS:
        try:
            with open(path, "rb") as f:
                print(f"Parsing {path}...")

                # The MNIST files where written in older architectures
                # like (SGI/Sun) who uses Big-Endian (the most significant
                # bytes comes first). Newer computers (AMD/Intel) normally
                # use Little-Endian.
                magic = struct.unpack(">I", f.read(4))[0]
                # Alternative using int
                # magic = int.from_bytes(f.read(4), byteorder="big", signed=False)
                if magic != 2049:
                    raise ValueError(
                        f"Invalid Magic Number: {magic}. Expected 2049 (0x00000801)."
                    )

                entries = struct.unpack(">I", f.read(4))[0]
                if entries not in [
                    10000,
                    60000,
                ]:  # [Train, Test] entries expected
                    raise ValueError(
                        f"Invalid entry count: {entries}. Expected 60000."
                    )

                byte_object = f.read()
                digit_frequencies = [0] * 10

                for label in byte_object:
                    if not 9 >= label >= 0:
                        raise ValueError(
                            f"Digits goes from 0 to 9 but label is {label}"
                        )
                    digit_frequencies[label] += 1

                print("\nParsing Result...")
                print(f"Magic Number: {magic}")
                print(f"Enstimated Entries: {entries}")
                print(f"Payload Bytes Read: {len(byte_object)}")
                for x, y in enumerate(digit_frequencies):
                    print(f"{x}", end=" ")
                    for i in range(0, int(y / 500)):
                        print("█", end="")
                    print(f" {y} labels")
                print(f"Total Entries: {sum(digit_frequencies)}")
        except FileNotFoundError:
            print("File was not found.")


if __name__ == "__main__":
    main()
