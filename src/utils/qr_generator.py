import qrcode
from qrcode.constants import ERROR_CORRECT_L
from PIL import Image
import io
import base64
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

class QRGenerator:
    """QR code generator with customization options."""
    
    def __init__(self, 
                 version: int = 1,
                 error_correction: int = ERROR_CORRECT_L,
                 box_size: int = 10,
                 border: int = 4,
                 fill_color: str = "black",
                 back_color: str = "white"):
        """
        Initialize QR code generator with customization options.
        
        Args:
            version: QR code version (1-40)
            error_correction: Error correction level
            box_size: Size of each box in pixels
            border: Border size in boxes
            fill_color: Color of QR code
            back_color: Background color
        """
        self.version = version
        self.error_correction = error_correction
        self.box_size = box_size
        self.border = border
        self.fill_color = fill_color
        self.back_color = back_color
        
    def generate(self, 
                data: str,
                logo_path: Optional[str] = None,
                logo_size: Optional[Tuple[int, int]] = None) -> Image.Image:
        """
        Generate QR code with optional logo.
        
        Args:
            data: Data to encode in QR code
            logo_path: Optional path to logo image
            logo_size: Optional size for logo (width, height)
            
        Returns:
            PIL Image object
        """
        try:
            # Create QR code
            qr = qrcode.QRCode(
                version=self.version,
                error_correction=self.error_correction,
                box_size=self.box_size,
                border=self.border
            )
            
            qr.add_data(data)
            qr.make(fit=True)
            
            # Create image
            qr_image = qr.make_image(
                fill_color=self.fill_color,
                back_color=self.back_color
            )
            
            # Add logo if provided
            if logo_path:
                try:
                    logo = Image.open(logo_path)
                    if logo_size:
                        logo = logo.resize(logo_size)
                    
                    # Calculate logo position
                    pos = ((qr_image.size[0] - logo.size[0]) // 2,
                          (qr_image.size[1] - logo.size[1]) // 2)
                    
                    # Paste logo
                    qr_image.paste(logo, pos)
                except Exception as e:
                    logger.error(f"Error adding logo to QR code: {str(e)}")
            
            return qr_image
            
        except Exception as e:
            logger.error(f"Error generating QR code: {str(e)}")
            raise
            
    def generate_base64(self, 
                       data: str,
                       logo_path: Optional[str] = None,
                       logo_size: Optional[Tuple[int, int]] = None) -> str:
        """
        Generate QR code and return as base64 string.
        
        Args:
            data: Data to encode in QR code
            logo_path: Optional path to logo image
            logo_size: Optional size for logo (width, height)
            
        Returns:
            Base64 encoded string
        """
        try:
            qr_image = self.generate(data, logo_path, logo_size)
            
            # Convert to base64
            buffered = io.BytesIO()
            qr_image.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode()
            
        except Exception as e:
            logger.error(f"Error generating base64 QR code: {str(e)}")
            raise
            
    def generate_file(self,
                     data: str,
                     output_path: str,
                     logo_path: Optional[str] = None,
                     logo_size: Optional[Tuple[int, int]] = None) -> None:
        """
        Generate QR code and save to file.
        
        Args:
            data: Data to encode in QR code
            output_path: Path to save QR code
            logo_path: Optional path to logo image
            logo_size: Optional size for logo (width, height)
        """
        try:
            qr_image = self.generate(data, logo_path, logo_size)
            qr_image.save(output_path)
            
        except Exception as e:
            logger.error(f"Error saving QR code to file: {str(e)}")
            raise
            
    def validate_data(self, data: str) -> bool:
        """
        Validate data for QR code generation.
        
        Args:
            data: Data to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            qr = qrcode.QRCode(version=1)
            qr.add_data(data)
            qr.make(fit=True)
            return True
        except Exception:
            return False
            
    def get_qr_size(self, data: str) -> Tuple[int, int]:
        """
        Get size of QR code for given data.
        
        Args:
            data: Data to encode
            
        Returns:
            Tuple of (width, height) in pixels
        """
        try:
            qr = qrcode.QRCode(
                version=self.version,
                error_correction=self.error_correction,
                box_size=self.box_size,
                border=self.border
            )
            qr.add_data(data)
            qr.make(fit=True)
            return qr.get_matrix().size
        except Exception as e:
            logger.error(f"Error calculating QR code size: {str(e)}")
            raise 