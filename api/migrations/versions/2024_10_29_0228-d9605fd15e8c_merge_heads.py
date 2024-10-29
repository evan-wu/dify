"""merge heads

Revision ID: d9605fd15e8c
Revises: 334092df4bf7, bbadea11becb
Create Date: 2024-10-29 02:28:37.315224

"""
from alembic import op
import models as models
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'd9605fd15e8c'
down_revision = ('334092df4bf7', 'bbadea11becb')
branch_labels = None
depends_on = None


def upgrade():
    pass


def downgrade():
    pass
