use byteorder::{BigEndian, ByteOrder, LittleEndian};
use dicom::core::value::Value;
use dicom::{
    encoding::transfer_syntax::{Endianness, TransferSyntaxIndex},
    object::DefaultDicomObject,
    transfer_syntax::TransferSyntaxRegistry,
};
use gdcm_rs::{decode_single_frame_compressed, GDCMPhotometricInterpretation, GDCMTransferSyntax};
use image::{DynamicImage, ImageBuffer, Luma};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use snafu::OptionExt;
use snafu::{ResultExt, Snafu};
use std::str::FromStr;

#[derive(Debug, Snafu)]
pub enum Error {
    #[snafu(display("Missing required element"))]
    MissingRequiredField { source: dicom::object::Error },
    #[snafu(display("Could not parse element"))]
    ParseError {
        source: dicom::core::value::CastValueError,
    },
    #[snafu(display("Non supported GDCM PhotometricInterpretation {}", message))]
    GDCMNonSupportedPI {
        source: gdcm_rs::InvalidGDCMPI,
        message: String,
    },
    #[snafu(display("Non supported GDCM TransferSyntax {}", message))]
    GDCMNonSupportedTS {
        source: gdcm_rs::InvalidGDCMTS,
        message: String,
    },
    #[snafu(display("Non supported TransferSyntax {}", message))]
    NonSupportedTS { message: String },
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Decoded pixel data
/// NOTE: Is it possible to use dicom::PixelData ?
pub struct DecodedPixelData {
    data: Vec<u8>, // TODO: Make this a generic collection?
    rows: u32,
    cols: u32,
    endianness: Endianness,
    photometric_interpretation: String,
    samples_per_pixel: u16,
    bits_allocated: u16,
    bits_stored: u16,
    high_bit: u16,
    pixel_representation: u16,
}

impl DecodedPixelData {
    /// Convert decoded pixel data into a DynamicImage
    /// Please note that many settings are not yet supported!
    /// Such as:
    ///  - Apply HU
    ///  - Apply WindowWidth/WindowHeight
    ///  - Convert PhotometricInterpretation
    fn to_dynamic_image(&self) -> Result<DynamicImage> {
        if self.photometric_interpretation != "MONOCHROME2" {
            todo!("Only MONOCHROME2 is supported as photometric interpretation")
        }

        match self.bits_allocated {
            8 => match self.samples_per_pixel {
                1 => {
                    let image_buffer: ImageBuffer<Luma<u8>, Vec<u8>> =
                        ImageBuffer::from_raw(self.cols, self.rows, self.data.to_owned()).unwrap();
                    return Ok(DynamicImage::from(DynamicImage::ImageLuma8(image_buffer)));
                }
                _ => {
                    todo!("RGB is not yet supported");
                }
            },
            16 => {
                let mut dest = vec![0; self.data.len() / 2];
                match self.pixel_representation {
                    0 => {
                        // TODO: This is going wrong
                        // Somehow, the endianness doesn't always reflect the real endianness
                        // Try this on Linux with machinne format, maybe we don't need to convert it

                        // Unsigned 16 bit data, lookup pixel data storage order
                        match self.endianness {
                            // NOTE: Is it possible to use the macro here for read? Instead of another match?
                            Endianness::Little => {
                                println!("CONV LITTLE");
                                LittleEndian::read_u16_into(&self.data, &mut dest);
                            }
                            Endianness::Big => {
                                println!("CONV BIG");
                                BigEndian::read_u16_into(&self.data, &mut dest);
                            }
                        }
                    }
                    1 => {
                        // Signed 16 bit data in 2s complement
                        let mut signed_buffer = vec![0; self.data.len() / 2];
                        match self.endianness {
                            Endianness::Little => {
                                LittleEndian::read_i16_into(&self.data, &mut signed_buffer);
                            }
                            Endianness::Big => {
                                BigEndian::read_i16_into(&self.data, &mut signed_buffer);
                            }
                        }

                        // TODO: This is a dirty way to rescale pixels to u16
                        // TODO: Figure out how this should be done properly
                        // TODO: I implemented this in a rage
                        let max = *signed_buffer.iter().max().unwrap() as f32;
                        dest = signed_buffer
                            .par_iter()
                            .map(|v| {
                                if *v < 0 {
                                    return 0;
                                }
                                let mut rescaled_value = *v as f32;
                                rescaled_value = (rescaled_value / max) * (i16::MAX as f32);
                                rescaled_value as u16
                            })
                            .collect();
                    }
                    _ => panic!("Invalid pixel representation {}", self.pixel_representation),
                }
                let image_buffer: ImageBuffer<Luma<u16>, Vec<u16>> =
                    ImageBuffer::from_raw(self.cols, self.rows, dest).unwrap();
                Ok(DynamicImage::from(DynamicImage::ImageLuma16(image_buffer)))
            }
            _ => panic!("BitsAllocated must be 8 or 16, not {}", self.bits_allocated),
        }
    }
}

pub trait PixelDecoder {
    /// Decode compressed pixel data
    fn decode_pixel_data(&self) -> Result<DecodedPixelData>;
}

impl PixelDecoder for DefaultDicomObject {
    /// Decode encapsulated pixel data
    fn decode_pixel_data(&self) -> Result<DecodedPixelData> {
        let pixel_data = self
            .element_by_name("PixelData")
            .context(MissingRequiredField)?;
        let cols = get_cols(self)?;
        let rows = get_rows(self)?;

        let photometric_interpretation = get_photometric_interpretation(self)?;
        let pi_type = GDCMPhotometricInterpretation::from_str(&photometric_interpretation)
            .context(GDCMNonSupportedPI {
                message: &photometric_interpretation,
            })?;

        let transfer_syntax = &self.meta().transfer_syntax;
        let registry = TransferSyntaxRegistry
            .get(&&transfer_syntax)
            .context(NonSupportedTS {
                message: transfer_syntax,
            })?;
        let endianness = registry.endianness();
        println!("ENDIANNESS {:?}", endianness);
        println!("TS {}", transfer_syntax);
        let ts_type =
            GDCMTransferSyntax::from_str(&registry.uid()).context(GDCMNonSupportedTS {
                message: transfer_syntax,
            })?;

        let samples_per_pixel = get_samples_per_pixel(self)?;
        let bits_allocated = get_bits_allocated(self)?;
        let bits_stored = get_bits_stored(self)?;
        let high_bit = get_high_bit(self)?;
        let pixel_representation = get_pixel_representation(self)?;

        let decoded_pixel_data = match pixel_data.value() {
            Value::PixelSequence {
                fragments,
                offset_table: _,
            } => {
                if fragments.len() > 1 {
                    // Bundle fragments and decode multi-frame dicoms
                    todo!("Not yet implemented");
                }
                let decoded_frame = decode_single_frame_compressed(
                    &fragments[0],
                    cols.into(),
                    rows.into(),
                    pi_type,
                    ts_type,
                    samples_per_pixel,
                    bits_allocated,
                    bits_stored,
                    high_bit,
                    pixel_representation,
                );
                decoded_frame.unwrap().to_vec()
            }
            Value::Primitive(p) => {
                // Non-encoded, just return the pixel data
                p.to_bytes().to_vec()
            }
            Value::Sequence { items: _, size: _ } => {
                todo!("Not yet implemented");
            }
        };

        return Ok(DecodedPixelData {
            data: decoded_pixel_data,
            cols: cols.into(),
            rows: rows.into(),
            endianness,
            photometric_interpretation,
            samples_per_pixel,
            bits_allocated,
            bits_stored,
            high_bit,
            pixel_representation,
        });
    }
}

/// Get the width of the dicom
fn get_cols(obj: &DefaultDicomObject) -> Result<u16> {
    Ok(obj
        .element_by_name("Columns")
        .context(MissingRequiredField)?
        .uint16()
        .context(ParseError)?)
}

/// Get the height of the dicom
fn get_rows(obj: &DefaultDicomObject) -> Result<u16> {
    Ok(obj
        .element_by_name("Rows")
        .context(MissingRequiredField)?
        .uint16()
        .context(ParseError)?)
}

/// Get the PhotoMetricInterpretation
fn get_photometric_interpretation(obj: &DefaultDicomObject) -> Result<String> {
    Ok(obj
        .element_by_name("PhotometricInterpretation")
        .context(MissingRequiredField)?
        .string()
        .context(ParseError)?
        .trim()
        .to_string())
}

/// Get the SamplesPerPixel
fn get_samples_per_pixel(obj: &DefaultDicomObject) -> Result<u16> {
    Ok(obj
        .element_by_name("SamplesPerPixel")
        .context(MissingRequiredField)?
        .uint16()
        .context(ParseError)?)
}

/// Get the BitsAllocated
fn get_bits_allocated(obj: &DefaultDicomObject) -> Result<u16> {
    Ok(obj
        .element_by_name("BitsAllocated")
        .context(MissingRequiredField)?
        .uint16()
        .context(ParseError)?)
}

/// Get the BitsStored
fn get_bits_stored(obj: &DefaultDicomObject) -> Result<u16> {
    Ok(obj
        .element_by_name("BitsStored")
        .context(MissingRequiredField)?
        .uint16()
        .context(ParseError)?)
}

/// Get the HighBit
fn get_high_bit(obj: &DefaultDicomObject) -> Result<u16> {
    Ok(obj
        .element_by_name("HighBit")
        .context(MissingRequiredField)?
        .uint16()
        .context(ParseError)?)
}

/// Get the PixelRepresentation
fn get_pixel_representation(obj: &DefaultDicomObject) -> Result<u16> {
    Ok(obj
        .element_by_name("PixelRepresentation")
        .context(MissingRequiredField)?
        .uint16()
        .context(ParseError)?)
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use dicom::object::open_file;
    use dicom_test_files;
    use rstest::rstest;

    #[rstest(value => [
        // "pydicom/693_J2KI.dcm",
        // "pydicom/693_J2KR.dcm",
        // "pydicom/693_UNCI.dcm",
        // "pydicom/693_UNCR.dcm",
        // "pydicom/CT_small.dcm",
        // "pydicom/ExplVR_BigEnd.dcm",
        // "pydicom/ExplVR_BigEndNoMeta.dcm",
        // "pydicom/ExplVR_LitEndNoMeta.dcm",
        // "pydicom/JPEG-LL.dcm",               // More than 1 fragment
         "pydicom/JPEG-lossy.dcm",
         // "pydicom/JPEG2000.dcm",
        //  "pydicom/JPEG2000_UNC.dcm",
        //  "pydicom/JPGLosslessP14SV1_1s_1f_8b.dcm",

        // "pydicom/MR-SIEMENS-DICOM-WithOverlays.dcm",
        // "pydicom/MR2_J2KI.dcm",
        // "pydicom/MR2_J2KR.dcm",
        // "pydicom/MR2_UNCI.dcm",
        // "pydicom/MR2_UNCR.dcm",
        // "pydicom/MR_small.dcm",
        // "pydicom/MR_small_RLE.dcm",
        // "pydicom/MR_small_bigendian.dcm",
        // "pydicom/MR_small_expb.dcm",
        // "pydicom/MR_small_implicit.dcm",
        // "pydicom/MR_small_jp2klossless.dcm",
        // "pydicom/MR_small_jpeg_ls_lossless.dcm",
        // "pydicom/MR_small_padded.dcm",
        // "pydicom/MR_truncated.dcm",
        // "pydicom/OBXXXX1A.dcm",
        // "pydicom/OBXXXX1A_2frame.dcm",
        // "pydicom/OBXXXX1A_expb.dcm",
        // "pydicom/OBXXXX1A_expb_2frame.dcm",
        // "pydicom/OBXXXX1A_rle.dcm",
        // "pydicom/OBXXXX1A_rle_2frame.dcm",
        // "pydicom/OT-PAL-8-face.dcm",
        // "pydicom/README.txt",
        // "pydicom/RG1_J2KI.dcm",
        // "pydicom/RG1_J2KR.dcm",
        // "pydicom/RG1_UNCI.dcm",
        // "pydicom/RG1_UNCR.dcm",
        // "pydicom/RG3_J2KI.dcm",
        // "pydicom/RG3_J2KR.dcm",
        // "pydicom/RG3_UNCI.dcm",
        // "pydicom/RG3_UNCR.dcm",
        // "pydicom/SC_rgb.dcm",
        // "pydicom/SC_rgb_16bit.dcm",
        // "pydicom/SC_rgb_16bit_2frame.dcm",
        // "pydicom/SC_rgb_2frame.dcm",
        // "pydicom/SC_rgb_32bit.dcm",
        // "pydicom/SC_rgb_32bit_2frame.dcm",
        // "pydicom/SC_rgb_dcmtk_+eb+cr.dcm",
        // "pydicom/SC_rgb_dcmtk_+eb+cy+n1.dcm",
        // "pydicom/SC_rgb_dcmtk_+eb+cy+n2.dcm",
        // "pydicom/SC_rgb_dcmtk_+eb+cy+np.dcm",
        // "pydicom/SC_rgb_dcmtk_+eb+cy+s2.dcm",
        // "pydicom/SC_rgb_dcmtk_+eb+cy+s4.dcm",
        // "pydicom/SC_rgb_dcmtk_ebcr_dcmd.dcm",
        // "pydicom/SC_rgb_dcmtk_ebcyn1_dcmd.dcm",
        // "pydicom/SC_rgb_dcmtk_ebcyn2_dcmd.dcm",
        // "pydicom/SC_rgb_dcmtk_ebcynp_dcmd.dcm",
        // "pydicom/SC_rgb_dcmtk_ebcys2_dcmd.dcm",
        // "pydicom/SC_rgb_dcmtk_ebcys4_dcmd.dcm",
        // "pydicom/SC_rgb_expb.dcm",
        // "pydicom/SC_rgb_expb_16bit.dcm",
        // "pydicom/SC_rgb_expb_16bit_2frame.dcm",
        // "pydicom/SC_rgb_expb_2frame.dcm",
        // "pydicom/SC_rgb_expb_32bit.dcm",
        // "pydicom/SC_rgb_expb_32bit_2frame.dcm",
        // "pydicom/SC_rgb_gdcm2k_uncompressed.dcm",
        // "pydicom/SC_rgb_gdcm_KY.dcm",
        // "pydicom/SC_rgb_jpeg_dcmtk.dcm",
        // "pydicom/SC_rgb_jpeg_gdcm.dcm",
        // "pydicom/SC_rgb_jpeg_lossy_gdcm.dcm",
        // "pydicom/SC_rgb_rle.dcm",
        // "pydicom/SC_rgb_rle_16bit.dcm",
        // "pydicom/SC_rgb_rle_16bit_2frame.dcm",
        // "pydicom/SC_rgb_rle_2frame.dcm",
        // "pydicom/SC_rgb_rle_32bit.dcm",
        // "pydicom/SC_rgb_rle_32bit_2frame.dcm",
        // "pydicom/SC_rgb_small_odd.dcm",
        // "pydicom/SC_rgb_small_odd_jpeg.dcm",
        // "pydicom/SC_ybr_full_422_uncompressed.dcm",
        // "pydicom/SC_ybr_full_uncompressed.dcm",
        // "pydicom/US1_J2KI.dcm",
        // "pydicom/US1_J2KR.dcm",
        // "pydicom/US1_UNCI.dcm",
        // "pydicom/US1_UNCR.dcm",
        // "pydicom/badVR.dcm",
        // "pydicom/bad_sequence.dcm",
        // "pydicom/color-pl.dcm",
        // "pydicom/color-px.dcm",
        // "pydicom/color3d_jpeg_baseline.dcm",
        // "pydicom/eCT_Supplemental.dcm",
        // "pydicom/empty_charset_LEI.dcm",
        // "pydicom/emri_small.dcm",
        // "pydicom/emri_small_RLE.dcm",
        // "pydicom/emri_small_big_endian.dcm",
        // "pydicom/emri_small_jpeg_2k_lossless.dcm",
        // "pydicom/emri_small_jpeg_2k_lossless_too_short.dcm",
        // "pydicom/emri_small_jpeg_ls_lossless.dcm",
        // "pydicom/explicit_VR-UN.dcm",
        // "pydicom/gdcm-US-ALOKA-16.dcm",
        // "pydicom/gdcm-US-ALOKA-16_big.dcm",
        // "pydicom/image_dfl.dcm",
        // "pydicom/liver.dcm",
        // "pydicom/liver_1frame.dcm",
        // "pydicom/liver_expb.dcm",
        // "pydicom/liver_expb_1frame.dcm",
        // "pydicom/meta_missing_tsyntax.dcm",
        // "pydicom/mlut_18.dcm",
        // "pydicom/nested_priv_SQ.dcm",
        // "pydicom/no_meta.dcm",
        // "pydicom/no_meta_group_length.dcm",
        // "pydicom/priv_SQ.dcm",
        // "pydicom/reportsi.dcm",
        // "pydicom/reportsi_with_empty_number_tags.dcm",
        // "pydicom/rtdose.dcm",
        // "pydicom/rtdose_1frame.dcm",
        // "pydicom/rtdose_expb.dcm",
        // "pydicom/rtdose_expb_1frame.dcm",
        // "pydicom/rtdose_rle.dcm",
        // "pydicom/rtdose_rle_1frame.dcm",
        // "pydicom/rtplan.dcm",
        // "pydicom/rtplan_truncated.dcm",
        // "pydicom/rtstruct.dcm",
        // "pydicom/test-SR.dcm",
        // "pydicom/vlut_04.dcm",
    ])]
    fn test_parse_dicom_pixel_data(value: &str) {
        println!("Parsing pixel data for {}", value);

        let obj = open_file(dicom_test_files::path(value).unwrap()).unwrap();
        let image = obj.decode_pixel_data().unwrap().to_dynamic_image().unwrap();
        image
            .save(format!(
                "target/{}.png",
                Path::new(value).file_stem().unwrap().to_str().unwrap()
            ))
            .unwrap();
    }
}
