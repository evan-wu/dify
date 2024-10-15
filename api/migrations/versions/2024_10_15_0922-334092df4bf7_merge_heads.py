"""merge heads

Revision ID: 334092df4bf7
Revises: d8e744d88ed6, 418bb48a90be
Create Date: 2024-10-15 09:22:31.835480

"""
from alembic import op
import models as models
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '334092df4bf7'
down_revision = ('d8e744d88ed6', '418bb48a90be')
branch_labels = None
depends_on = None


def upgrade():
    pass


def downgrade():
    pass
